# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KjOgju_A5qAfmPtAI9ieAhWj_bZdodeY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the function to fuse LoRA weights
def fuse_lora_weights(loraA, loraB):
    """
    Fuse LoRA weights into a single convolutional kernel.

    Args:
    loraA (torch.Tensor): Weights of LoraA with shape (lora_rank, in_channels, 3, 3)
    loraB (torch.Tensor): Weights of LoraB with shape (out_channels, lora_rank, 1, 1)

    Returns:
    torch.Tensor: Fused weights with shape (out_channels, in_channels, 3, 3)
    """
    out_channels, lora_rank, _, _ = loraB.shape
    lora_rank, in_channels, kh, kw = loraA.shape

    # Initialize the fused weight
    fused_weight = torch.zeros((out_channels, in_channels, kh, kw))

    # Perform the weight fusion
    for i in range(out_channels):
        for j in range(lora_rank):
            fused_weight[i] += loraB[i, j] * loraA[j]

    return fused_weight

# Example LoRA parameters
in_channels = 3
out_channels = 16
lora_rank = 8

# Randomly initialize LoRA weights
loraA = torch.randn(lora_rank, in_channels, 3, 3)
loraB = torch.randn(out_channels, lora_rank, 1, 1)

# Create input tensor
input_tensor = torch.randn(1, in_channels, 64, 64)  # Example input tensor

# Sequential application of LoRA layers
loraA_layer = nn.Conv2d(in_channels, lora_rank, kernel_size=3, padding=1, bias=False)
loraB_layer = nn.Conv2d(lora_rank, out_channels, kernel_size=1, bias=False)

# Set the weights of the LoRA layers
loraA_layer.weight.data = loraA
loraB_layer.weight.data = loraB

# Forward pass through sequential LoRA layers
output_sequential = loraB_layer(loraA_layer(input_tensor))

# Fused weights application
fused_weight = fuse_lora_weights(loraA, loraB)
fused_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
fused_layer.weight.data = fused_weight

# Forward pass through the fused layer
output_fused = fused_layer(input_tensor)

# Compare the outputs
print("Difference between sequential and fused outputs:", torch.abs(output_sequential - output_fused).max().item())

import torch
import torch.nn as nn
import torch.nn.functional as F

def fuse_lora_weights_einsum(loraA, loraB):
    """
    Fuse LoRA weights into a single convolutional kernel using einsum.

    Args:
    loraA (torch.Tensor): Weights of LoraA with shape (lora_rank, in_channels, kh, kw)
    loraB (torch.Tensor): Weights of LoraB with shape (out_channels, lora_rank, 1, 1)

    Returns:
    torch.Tensor: Fused weights with shape (out_channels, in_channels, kh, kw)
    """
    # Use einsum to perform the weight fusion
    fused_weight = torch.einsum('oi,ijkl->ojkl', loraB.squeeze(-1).squeeze(-1), loraA)
    return fused_weight

# Example usage:
in_channels = 3
out_channels = 16
lora_rank = 8

# Randomly initialize LoRA weights
loraA = torch.randn(lora_rank, in_channels, 3, 3)
loraB = torch.randn(out_channels, lora_rank, 1, 1)

# Fuse the weights
fused_weight = fuse_lora_weights_einsum(loraA, loraB)
print(fused_weight.shape)  # Should be (out_channels, in_channels, 3, 3)

# Create input tensor
input_tensor = torch.randn(1, in_channels, 64, 64)  # Example input tensor

# Sequential application of LoRA layers
loraA_layer = nn.Conv2d(in_channels, lora_rank, kernel_size=3, padding=1, bias=False)
loraB_layer = nn.Conv2d(lora_rank, out_channels, kernel_size=1, bias=False)

# Set the weights of the LoRA layers
loraA_layer.weight.data = loraA
loraB_layer.weight.data = loraB

# Forward pass through sequential LoRA layers
output_sequential = loraB_layer(loraA_layer(input_tensor))

# Fused weights application
fused_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
fused_layer.weight.data = fused_weight

# Forward pass through the fused layer
output_fused = fused_layer(input_tensor)

# Compare the outputs
print("Difference between sequential and fused outputs:", torch.abs(output_sequential - output_fused).max().item())

