from typing import Dict, List, Literal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm, trange

from colorization.colorizers import (
    ECCVGenerator,
    SIGGRAPHGenerator,
    eccv16,
    siggraph17,
)  # your existing factory function
from src.LoRA import LoRAConv2d, inject_lora
from src.model_saver import ModelSaver

# ── 1. Freeze encoder (model1-6), keep decoder trainable ─────────────────────

"""
def build_model(
    model_name: Literal["eccv16", "siggraph17"],
    target_layers: List[str],
    pretrained: bool = True,
) -> ECCVGenerator | SIGGRAPHGenerator:
    if model_name == "eccv16":
        model = eccv16(pretrained=pretrained)
    elif model_name == "siggraph17":
        model = siggraph17(pretrained=pretrained)
    else:
        raise ValueError("model_name must be one of eccv16 or siggraph17")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the decoder head
    for name in target_layers:
        module = getattr(model, name)  # resolves 'model7' → model.model7
        for param in module.parameters():
            param.requires_grad = True

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {n_frozen:,}  |  Trainable: {n_trainable:,}")
    return model
"""


def build_model(
    model_name: Literal["eccv16", "siggraph17"],
    lora_layers: List[str] | None = None,
    finetune_layers: List[str] | None = None,
    pretrained: bool = True,
) -> ECCVGenerator | SIGGRAPHGenerator:
    if model_name == "eccv16":
        model = eccv16(pretrained=pretrained)
    elif model_name == "siggraph17":
        model = siggraph17(pretrained=pretrained)
    else:
        raise ValueError("model_name must be one of eccv16 or siggraph17")

    for param in model.parameters():
        param.requires_grad = False

    # Put entire model in eval mode (freezes BN stats)
    model.eval()

    # Then unfreeze + set train only the layers you're actually training
    if finetune_layers is not None:
        for name in finetune_layers:
            module = model.get_submodule(name)
            module.train()
            for param in module.parameters():
                param.requires_grad = True

    if lora_layers is not None:
        for name in lora_layers:
            inject_lora(model, target_name=name)

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {n_frozen:,}  |  Trainable: {n_trainable:,}")
    return model


def ab_to_soft_target(ab, pts, sigma=0.1):
    # ab: (B, 2, H, W), pts: (313, 2)
    B, _, H, W = ab.shape
    ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)
    dist_sq = ((ab_flat[:, None] - pts[None]) ** 2).sum(-1)  # (B*H*W, 313)
    target = F.softmax(-dist_sq / (2 * sigma**2), dim=1)
    return target.reshape(B, H, W, 313).permute(0, 3, 1, 2)


def image_log_likelihood_per_sample_soft(logits, ab_target, pts, sigma=0.1):
    soft_tgt = ab_to_soft_target(ab_target, pts, sigma)
    log_probs = F.log_softmax(logits, dim=1)
    ll_pixel = (soft_tgt * log_probs).sum(dim=1)  # (B, H, W)
    return ll_pixel.sum(dim=(1, 2))  # (B,)


# ── 2. DPO dataset ────────────────────────────────────────────────────────────
# Each sample: grayscale L input, a "chosen" ab output, a "rejected" ab output.
# Shape convention: (1, H, W) for L, (2, H, W) for ab — values in model space.


# custom type for the samples of the DPO training
# just to ease notation
ColorDPOSamples = List[Dict[Literal["l", "chosen_ab", "rejected_ab"], torch.Tensor]]


class ColorDPODataset(Dataset):
    def __init__(self, samples: ColorDPOSamples) -> None:
        # samples: list of dicts with keys 'l', 'chosen_ab', 'rejected_ab'
        self.samples = samples
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            s["l"].clone().detach().to(torch.float32),  # (1, H, W)
            s["chosen_ab"].clone().detach().to(torch.float32),  # (2, H, W)
            s["rejected_ab"].clone().detach().to(torch.float32),  # (2, H, W)
        )


# ── 3. DPO loss ───────────────────────────────────────────────────────────────
# Treat each pixel as an independent "token" emitting a 2-dim ab prediction.
# log-likelihood = -MSE (Gaussian policy); beta scales the KL penalty.


def image_log_likelihood_from_logits(
    logits: torch.Tensor, ab_target: torch.Tensor, pts: torch.Tensor
) -> torch.Tensor:
    """
    logits    : (B, 313, H, W) — raw model output before softmax
    ab_target : (B, 2, H, W)   — target ab image (normalized)
    pts       : (313, 2)        — bin centroids
    Returns scalar log-likelihood averaged over pixels.
    """
    B, _, H, W = logits.shape

    # Downsample target to match logits spatial size (H/4, W/4)
    ab_down = F.interpolate(
        ab_target, size=(H, W), mode="bilinear", align_corners=False
    )

    # Convert ab to nearest bin index
    bin_indices = ab_to_bin_indices(ab_down, pts)  # (B, H, W)

    # Cross-entropy = negative log-likelihood under categorical distribution
    log_probs = F.log_softmax(logits, dim=1)  # (B, 313, H, W)
    ll = -F.nll_loss(log_probs, bin_indices, reduction="sum")  # reduction="mean"
    return ll


def dpo_loss(
    policy_model: ECCVGenerator,
    ref_model: ECCVGenerator,
    l_batch: torch.Tensor,
    chosen_ab: torch.Tensor,
    rejected_ab: torch.Tensor,
    pts: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    with torch.no_grad():
        ref_logits = get_logits(ref_model, l_batch)
    pol_logits = get_logits(policy_model, l_batch)

    cab = chosen_ab / 110.0
    rab = rejected_ab / 110.0

    # Per-sample (B,) — somma sui pixel di una immagine, NIENTE somma sul batch
    ll_pol_c = image_log_likelihood_per_sample_soft(pol_logits, cab, pts)
    ll_pol_r = image_log_likelihood_per_sample_soft(pol_logits, rab, pts)
    ll_ref_c = image_log_likelihood_per_sample_soft(ref_logits, cab, pts)
    ll_ref_r = image_log_likelihood_per_sample_soft(ref_logits, rab, pts)

    log_ratio_c = ll_pol_c - ll_ref_c  # (B,)
    log_ratio_r = ll_pol_r - ll_ref_r  # (B,)

    # logsigmoid per-coppia, poi media — DPO standard
    loss = -F.logsigmoid(beta * (log_ratio_c - log_ratio_r))  # (B,)
    return loss.mean()


pts_in_hull = np.load("pts_in_hull.npy")
pts = torch.from_numpy(pts_in_hull).float() / 110.0

# ── 4. Training loop ──────────────────────────────────────────────────────────


def train(
    model: ECCVGenerator,
    ref_model: ECCVGenerator,
    lora_layers: List[str],
    finetune_layers: List[str],
    samples: ColorDPOSamples,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    beta: float = 0.1,
    test_size: float = 0.15,
    val_every: int = 5,
    saving_epochs: List[int] | None = None,
    **kwargs,
) -> ECCVGenerator:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- DATASETS INITIALIZATION ---
    assert 0.0 < test_size < 1.0, "test_size must be between 0 and 1"
    n_val = max(1, int(len(samples) * test_size))
    indices = torch.randperm(len(samples)).tolist()
    val_samples = [samples[i] for i in indices[:n_val]]
    train_samples = [samples[i] for i in indices[n_val:]]
    print(f"Dataset split — Train: {len(train_samples)}  |  Val: {len(val_samples)}")
    train_loader = DataLoader(
        ColorDPODataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        ColorDPODataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # --- MODELS INITIALIZATION ---
    policy_model = model.to(device).eval()
    for name in lora_layers + (finetune_layers or []):
        policy_model.get_submodule(name).train()  # only these go train
    ref_model = ref_model.to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    policy_model = model.to(device).eval()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-2
    )

    # --- MODEL SAVER INITIALIZATION ---
    index_path = Path("models") / "index.json"
    state_dicts_path = Path("models") / "state_dicts"
    model_saver = ModelSaver(index_path, state_dicts_path)

    with torch.no_grad():
        l, c, r = next(iter(train_loader))
        ref_logits = get_logits(ref_model, l.to(device))
        ll_chosen = image_log_likelihood_from_logits(
            ref_logits, c.to(device) / 110.0, pts.to(device)
        )
        ll_rejected = image_log_likelihood_from_logits(
            ref_logits, r.to(device) / 110.0, pts.to(device)
        )
        print(
            f"Ref LL chosen={ll_chosen:.4f}  rejected={ll_rejected:.4f}  gap={ll_chosen - ll_rejected:.4f}"
        )

    train_losses = []
    val_losses = []
    val_epochs = []
    for epoch in trange(epochs, desc="Epochs"):
        total_loss = 0.0
        batch_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False
        )

        for l_batch, chosen_ab, rejected_ab in batch_bar:
            l_batch = l_batch.to(device)
            chosen_ab = chosen_ab.to(device)
            rejected_ab = rejected_ab.to(device)

            optimizer.zero_grad()
            loss = dpo_loss(
                policy_model,
                ref_model,
                l_batch,
                chosen_ab,
                rejected_ab,
                pts=pts.to(device),
                beta=beta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── Validation epoch ──────────────────────────────────────────────────

        if (epoch + 1) % val_every == 0:
            val_epochs.append(epoch)
            policy_model.eval()

            print(
                f"Epoch {epoch + 1:>3}/{epochs}  train={avg_train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # accumulatori per-sample, non per-batch (così le medie sono pesate corrette)
            buf = {
                k: []
                for k in [
                    "loss",
                    "ll_pol_chosen",
                    "ll_pol_rej",
                    "ll_ref_chosen",
                    "ll_ref_rej",
                ]
            }

            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False
            )
            for l_batch, chosen_ab, rejected_ab in val_bar:
                l_batch = l_batch.to(device)
                chosen_ab = chosen_ab.to(device)
                rejected_ab = rejected_ab.to(device)

                out = dpo_val_step(
                    policy_model,
                    ref_model,
                    l_batch,
                    chosen_ab,
                    rejected_ab,
                    pts=pts.to(device),
                    beta=beta,
                )
                for k in buf:
                    buf[k].append(out[k].cpu())

            # concat e calcola scalari finali
            for k in buf:
                buf[k] = torch.cat(buf[k])  # ognuno è (N_val,)

            avg_val_loss = buf["loss"].mean().item()
            margin = (buf["ll_pol_chosen"] - buf["ll_pol_rej"]).mean().item()
            win_rate = (buf["ll_pol_chosen"] > buf["ll_pol_rej"]).float().mean().item()
            r_chosen = (
                beta * (buf["ll_pol_chosen"] - buf["ll_ref_chosen"]).mean().item()
            )
            r_rejected = beta * (buf["ll_pol_rej"] - buf["ll_ref_rej"]).mean().item()

            val_losses.append(avg_val_loss)
            print(
                f"Epoch {epoch + 1:>3}/{epochs}  "
                f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
                f"win={win_rate * 100:.1f}%  margin={margin:+.2f}  "
                f"r_c={r_chosen:+.3f}  r_r={r_rejected:+.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # alarm: drift in collasso
            if r_rejected < -5.0:
                print(
                    f"  ⚠️  r_rejected={r_rejected:.2f} sotto -5: possibile collasso, considera early stop o alzare β"
                )

        if saving_epochs is not None:
            if epoch + 1 in saving_epochs:
                model_saver.save_model(model, epochs=epoch, **kwargs)

        plot_losses(list(range(epochs)), val_epochs, train_losses, val_losses)

    return policy_model


def plot_losses(
    train_epochs: List[int],
    val_epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
) -> None:
    # ── Loss summary plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_epochs, train_losses, label="train", linewidth=1.5)
    ax.plot(val_epochs, val_losses, "o--", label="val", linewidth=1.5)
    ax.axhline(
        y=0.6931, color="grey", linestyle=":", linewidth=1, label="log(2) baseline"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DPO Loss")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


def get_logits(model, input_l):
    conv1_2 = model.model1(model.normalize_l(input_l))
    conv2_2 = model.model2(conv1_2)
    conv3_3 = model.model3(conv2_2)
    conv4_3 = model.model4(conv3_3)
    conv5_3 = model.model5(conv4_3)
    conv6_3 = model.model6(conv5_3)
    conv7_3 = model.model7(conv6_3)
    conv8_3 = model.model8(conv7_3)  # (B, 313, H/4, W/4) ← stop here
    return conv8_3


def ab_to_bin_indices(ab: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    ab   : (B, 2, H, W) — normalized ab values
    pts  : (313, 2)      — bin centroids
    Returns: (B, H, W)   — index of nearest bin per pixel
    """
    B, _, H, W = ab.shape
    ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)
    dists = torch.cdist(ab_flat, pts)  # (B*H*W, 313)
    indices = dists.argmin(dim=1).reshape(B, H, W)  # (B, H, W)
    return indices


def image_log_likelihood_per_sample(logits, ab_target, pts):
    """
    Restituisce (B,): log π(y|x) sommata sui pixel, un valore per sample.
    Diversa da image_log_likelihood_from_logits: nessuna media, restituisce
    un tensor per-batch invece di uno scalare.
    """
    B, _, H, W = logits.shape
    ab_down = F.interpolate(
        ab_target, size=(H, W), mode="bilinear", align_corners=False
    )
    bin_indices = ab_to_bin_indices(ab_down, pts)  # (B, H, W)
    log_probs = F.log_softmax(logits, dim=1)  # (B, 313, H, W)
    ll_pixel = log_probs.gather(1, bin_indices.unsqueeze(1)).squeeze(1)  # (B, H, W)
    return ll_pixel.sum(dim=(1, 2))  # (B,)


@torch.no_grad()
def dpo_val_step(policy_model, ref_model, l_batch, chosen_ab, rejected_ab, pts, beta):
    pol_logits = get_logits(policy_model, l_batch)
    ref_logits = get_logits(ref_model, l_batch)

    cab = chosen_ab / 110.0
    rab = rejected_ab / 110.0

    ll_pol_c = image_log_likelihood_per_sample(pol_logits, cab, pts)
    ll_pol_r = image_log_likelihood_per_sample(pol_logits, rab, pts)
    ll_ref_c = image_log_likelihood_per_sample(ref_logits, cab, pts)
    ll_ref_r = image_log_likelihood_per_sample(ref_logits, rab, pts)

    log_ratio_c = ll_pol_c - ll_ref_c
    log_ratio_r = ll_pol_r - ll_ref_r
    loss_per_sample = -F.logsigmoid(beta * (log_ratio_c - log_ratio_r))

    return {
        "loss": loss_per_sample,  # (B,)
        "ll_pol_chosen": ll_pol_c,
        "ll_pol_rej": ll_pol_r,
        "ll_ref_chosen": ll_ref_c,
        "ll_ref_rej": ll_ref_r,
    }
