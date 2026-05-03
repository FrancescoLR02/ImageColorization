import torch
import torch.nn as nn
import math


class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, r: int = 8, alpha: int = 16) -> None:
        """
        Wraps an existing nn.Conv2d layer with LoRA parameters.
        """
        super().__init__()
        self.base_conv = base_conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 1. Freeze the original convolutional layer
        self.base_conv.weight.requires_grad = False
        if self.base_conv.bias is not None:
            self.base_conv.bias.requires_grad = False

        # 2. Define LoRA A: spatial convolution (K x K), reduces channels to 'r'
        self.lora_A = nn.Conv2d(
            in_channels=base_conv.in_channels,
            out_channels=r,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=base_conv.groups,
            bias=False,  # LoRA layers generally do not use biases
        )

        # 3. Define LoRA B: 1x1 convolution, projects 'r' to 'out_channels'
        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.reset_parameters()
        return

    def reset_parameters(self) -> None:
        # Initialize A with Kaiming uniform (standard for convolutions)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # Initialize B with zeros.
        # This ensures the initial output is exactly the same as the frozen base layer.
        nn.init.zeros_(self.lora_B.weight)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard forward pass through frozen weights
        base_out = self.base_conv(x)

        # LoRA forward pass, scaled by (alpha / r)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling

        # Add the two together
        return base_out + lora_out


def inject_lora(
    model: nn.Module, r: int = 8, alpha: int = 16, target_name: str | None = None
) -> nn.Module:
    """
    Recursively finds nn.Conv2d layers and wraps them in LoRAConv2d.
    Handles nested layers like nn.Sequential perfectly.
    """
    for name, module in model.named_children():
        # 1. Check if the module is a Conv2d layer
        is_conv = isinstance(module, nn.Conv2d)

        # 2. Check if it matches our naming criteria (if we provided one)
        matches_name = (target_name is None) or (target_name in name)

        if is_conv and matches_name:
            # Swap the layer
            lora_layer = LoRAConv2d(module, r=r, alpha=alpha)
            setattr(model, name, lora_layer)
        else:
            # If it's not a Conv2d (e.g., it's an nn.Sequential),
            # recursively dig into it and check its children.
            inject_lora(module, r=r, alpha=alpha, target_name=target_name)

    return model
