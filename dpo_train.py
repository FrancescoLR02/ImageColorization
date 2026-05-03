from typing import Dict, List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm, trange
from peft import LoraConfig, get_peft_model

from colorization.colorizers import (
    ECCVGenerator,
    SIGGRAPHGenerator,
    eccv16,
    siggraph17,
)  # your existing factory function
from LoRA import LoRAConv2d, inject_lora

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

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # we might want to retrain some layers (e.g. the decoder head)
    if finetune_layers is not None:
        for name in finetune_layers:
            module = getattr(model, name)  # resolves 'model7' → model.model7
            for param in module.parameters():
                param.requires_grad = True

    # add the LoRA layers
    if lora_layers is not None:
        for name in lora_layers:
            inject_lora(model, target_name=name)

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {n_frozen:,}  |  Trainable: {n_trainable:,}")
    return model


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
    ll = -F.nll_loss(log_probs, bin_indices, reduction="mean")
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

    # Normalize ab inputs (model expects normalized values)
    chosen_ab_n = chosen_ab / 110.0
    rejected_ab_n = rejected_ab / 110.0

    log_ratio_chosen = image_log_likelihood_from_logits(
        pol_logits, chosen_ab_n, pts
    ) - image_log_likelihood_from_logits(ref_logits, chosen_ab_n, pts)
    log_ratio_rejected = image_log_likelihood_from_logits(
        pol_logits, rejected_ab_n, pts
    ) - image_log_likelihood_from_logits(ref_logits, rejected_ab_n, pts)

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss.mean()


pts_in_hull = np.load("pts_in_hull.npy")  # (313, 2)
pts = torch.from_numpy(pts_in_hull).float()  # (313, 2)

# ── 4. Training loop ──────────────────────────────────────────────────────────


def train(
    samples: ColorDPOSamples,
    model: ECCVGenerator,
    ref_model: ECCVGenerator,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    beta: float = 0.1,
) -> ECCVGenerator:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_model = model.to(device).train()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Only pass trainable params to the optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    loader = DataLoader(
        ColorDPODataset(samples), batch_size=batch_size, shuffle=True, num_workers=4
    )

    for epoch in trange(epochs, desc="Epochs"):
        total_loss = 0.0
        batch_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
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
                pts=pts,
                beta=beta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            # Update inner bar with current loss
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")
    return policy_model


def get_logits(model: ECCVGenerator, input_l: torch.Tensor) -> torch.Tensor:
    """
    Returns the raw 313-bin logits before the softmax+collapse head.
    Shape: (B, 313, H/4, W/4)
    """
    conv1_2 = model.model1(model.normalize_l(input_l))
    conv2_2 = model.model2(conv1_2)
    conv3_3 = model.model3(conv2_2)
    conv4_3 = model.model4(conv3_3)
    conv5_3 = model.model5(conv4_3)
    conv6_3 = model.model6(conv5_3)
    conv7_3 = model.model7(conv6_3)
    conv8_3 = model.model8(conv7_3)  # (B, 313, H/4, W/4)
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
