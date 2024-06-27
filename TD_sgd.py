import torch
import torch.optim as optim

# Define the function to fuse LoRA weights (as before)
def fuse_lora_weights(loraA, loraB):
    out_channels, lora_rank, _, _ = loraB.shape
    lora_rank, in_channels, kh, kw = loraA.shape

    fused_weight = torch.zeros((out_channels, in_channels, kh, kw), device=loraA.device)

    for i in range(out_channels):
        for j in range(lora_rank):
            fused_weight[i] += loraB[i, j] * loraA[j]

    return fused_weight

# Given fused weights (for example, initialized randomly)
fused_weight_target = torch.randn(out_channels, in_channels, 3, 3, requires_grad=False)

# Initialize LoRA weights randomly
loraA = torch.randn(lora_rank, in_channels, 3, 3, requires_grad=True)
loraB = torch.randn(out_channels, lora_rank, 1, 1, requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([loraA, loraB], lr=0.01)

# Define the loss function (mean squared error)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Fuse LoRA weights
    fused_weight = fuse_lora_weights(loraA, loraB)
    
    # Compute the loss
    loss = loss_fn(fused_weight, fused_weight_target)
    
    # Backpropagation
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print the loss for every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# The optimized weights for LoRA A and LoRA B are now in loraA and loraB
