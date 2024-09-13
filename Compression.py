import torch
import torch.nn as nn
import numpy as np

class EvoConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, original_weights):
        super(EvoConvBlock, self).__init__()
        # Convolutional layer whose weights will be evolved
        self.conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.pointwise_conv = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize conv1 with weights from the original ResNet model
        self.conv.weight = nn.Parameter(original_weights[:out_channels // 2, :, :, :].clone())

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def set_conv_weights(self, weights):
        self.conv.weight = nn.Parameter(weights)
def modify_resnet(resnet):
    def replace_conv_layer(module, original_weights):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                # Replace the layer with EvoConvBlock
                new_layer = EvoConvBlock(in_channels, out_channels, kernel_size[0], stride[0], padding[0], original_weights)
                setattr(module, name, new_layer)
            elif len(list(layer.children())) > 0:
                replace_conv_layer(layer, original_weights)
    
    # Replace all convolution layers in ResNet
    original_weights = resnet.conv1.weight.data.clone()  # Extract original weights
    replace_conv_layer(resnet, original_weights)
    return resnet

# Load ResNet and modify it
resnet = models.resnet18(pretrained=True)
resnet = modify_resnet(resnet)
def evolutionary_algorithm(current_weights, fitness_function, population_size=10, mutation_rate=0.01, generations=10):
    population = [current_weights + mutation_rate * torch.randn_like(current_weights) for _ in range(population_size)]
    for generation in range(generations):
        fitness_scores = [fitness_function(weights) for weights in population]
        best_idx = np.argmax(fitness_scores)
        best_weights = population[best_idx]
        population = [best_weights + mutation_rate * torch.randn_like(best_weights) for _ in range(population_size)]
    return best_weights
def fitness_function(weights):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
# Training loop
model.to(device)
optimizer = optim.Adam([param for name, param in model.named_parameters() if 'pointwise_conv' in name], lr=0.001)  # Train only pointwise convs

for epoch in range(10):  # Example for 10 epochs
    # Update conv1 weights using the evolutionary algorithm
    for name, layer in model.named_modules():
        if isinstance(layer, EvoConvBlock):
            best_weights = evolutionary_algorithm(layer.conv.weight.data, fitness_function)
            layer.set_conv_weights(best_weights)
    
    # Train the pointwise convolutions
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
