import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion U-Net
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Step 1: Define LoRA configuration for the attention layers
lora_config_attention = LoraConfig(
    r=4,                # LoRA rank
    lora_alpha=16,      # Scaling factor for LoRA updates
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # LoRA applied to attention layers
    lora_dropout=0.05,  # Dropout rate for LoRA layers
    bias="none",        # No bias for LoRA layers
    task_type="stable-diffusion"  # Task-specific configuration
)

# Step 2: Apply LoRA to the U-Net's attention layers
pipe.unet = get_peft_model(pipe.unet, lora_config_attention)

# Define a custom LoRA layer for Conv2d
class CustomLoraConv2d(nn.Module):
    def __init__(self, conv_layer, rank=4, alpha=16):
        super(CustomLoraConv2d, self).__init__()
        self.conv = conv_layer  # Original Conv2d layer

        # LoRA decomposition
        self.lora_left = nn.Conv2d(conv_layer.in_channels, rank, kernel_size=1, bias=False)
        self.lora_right = nn.Conv2d(rank, conv_layer.out_channels, kernel_size=1, bias=False)

        # Initialize LoRA weights
        nn.init.zeros_(self.lora_left.weight)
        nn.init.zeros_(self.lora_right.weight)

        self.alpha = alpha

    def forward(self, x):
        # Original convolution
        out = self.conv(x)

        # LoRA residual
        lora_residual = self.lora_right(self.lora_left(x)) * self.alpha

        # Add LoRA residual to original output
        return out + lora_residual

# Apply the custom LoRA layer to the Conv2d layers in the U-Net
def apply_custom_lora_to_convs(unet, rank=4, alpha=16):
    for name, module in unet.named_modules():
        if isinstance(module, nn.Conv2d):
            custom_lora_layer = CustomLoraConv2d(module, rank=rank, alpha=alpha)
            parent, attr = name.rsplit('.', 1)
            setattr(getattr(unet, parent), attr, custom_lora_layer)

# Apply custom LoRA to convolutional layers
apply_custom_lora_to_convs(pipe.unet)
