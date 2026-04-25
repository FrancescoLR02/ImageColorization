# dpo_train.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from colorization import *  # your existing factory function
from tqdm import tqdm


import torch
from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model


# ── 1. Freeze encoder (model1-6), keep decoder trainable ─────────────────────

def build_model(model_name, target_layers ,pretrained=True):
    if model_name == 'eccv16':
        model = eccv16(pretrained=pretrained)
    if model_name == 'siggraph17':
        model = siggraph17(pretrained=pretrained)
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the decoder head
    for name in target_layers:
        module = getattr(model, name)  # resolves 'model7' → model.model7
        for param in module.parameters():
            param.requires_grad = True

    n_frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {n_frozen:,}  |  Trainable: {n_trainable:,}")
    return model


# ── 2. DPO dataset ────────────────────────────────────────────────────────────
# Each sample: grayscale L input, a "chosen" ab output, a "rejected" ab output.
# Shape convention: (1, H, W) for L, (2, H, W) for ab — values in model space.

class ColorDPODataset(Dataset):
    def __init__(self, samples):
        # samples: list of dicts with keys 'l', 'chosen_ab', 'rejected_ab'
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["l"].clone().detach().to(torch.float32),  # (1, H, W)
            s["chosen_ab"].clone().detach().to(torch.float32),  # (2, H, W)
            s["rejected_ab"].clone().detach().to(torch.float32),  # (2, H, W)
        )


# ── 3. DPO loss ───────────────────────────────────────────────────────────────
# Treat each pixel as an independent "token" emitting a 2-dim ab prediction.
# log-likelihood = -MSE (Gaussian policy); beta scales the KL penalty.

def image_log_likelihood(pred_ab, target_ab, sigma=0.1):
    """
    Gaussian log-likelihood over pixels.
    pred_ab, target_ab: (B, 2, H, W)
    Returns scalar per batch.
    """
    return -F.mse_loss(pred_ab, target_ab, reduction="mean") / (2 * sigma ** 2)

def dpo_loss(policy_model, ref_model, l_batch, chosen_ab, rejected_ab, beta=0.1, sigma=0.1):
    with torch.no_grad():
        ref_out = ref_model(l_batch)  # only one forward pass needed

    pol_chosen   = policy_model(l_batch)
    pol_rejected = policy_model(l_batch)

    out_h, out_w = pol_chosen.shape[2], pol_chosen.shape[3]
    chosen_ab_rs   = F.interpolate(chosen_ab,   size=(out_h, out_w), mode='bilinear', align_corners=False)
    rejected_ab_rs = F.interpolate(rejected_ab, size=(out_h, out_w), mode='bilinear', align_corners=False)

    log_ratio_chosen = (
        image_log_likelihood(pol_chosen,  chosen_ab_rs,   sigma) -
        image_log_likelihood(ref_out,     chosen_ab_rs,   sigma)
    )
    log_ratio_rejected = (
        image_log_likelihood(pol_rejected, rejected_ab_rs, sigma) -
        image_log_likelihood(ref_out,      rejected_ab_rs, sigma)
    )

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()
    return loss


# ── 4. Training loop ──────────────────────────────────────────────────────────

def train(samples, model, ref_model, epochs=5, batch_size=8, lr=1e-4, beta=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_model = model.to(device).train()
    #ref_model    = build_model(pretrained=True).to(device).eval()  # frozen reference
    for p in ref_model.parameters():
        p.requires_grad = False

    # Only pass trainable params to the optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    loader = DataLoader(
        ColorDPODataset(samples),
        batch_size=batch_size, shuffle=True, num_workers=4
    )

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        for l_batch, chosen_ab, rejected_ab in loader:
            l_batch    = l_batch.to(device)
            chosen_ab  = chosen_ab.to(device)
            rejected_ab = rejected_ab.to(device)

            optimizer.zero_grad()
            loss = dpo_loss(
                policy_model, ref_model,
                l_batch, chosen_ab, rejected_ab,
                beta=beta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    return policy_model