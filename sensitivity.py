import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN architecture
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a function to compute the sensitivity of each neuron in a layer
def compute_sensitivity(model, inputs):
    sensitivities = []
    for i in range(len(model)):
        layer = model[i]
        if isinstance(layer, nn.Conv2d):
            sensitivity = torch.zeros(layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1])
            for j in range(layer.out_channels):
                output = layer(inputs)
                loss = -output[:, j].sum()
                loss.backward(retain_graph=True)
                sensitivity[j] = layer.weight.grad.abs().mean(dim=(2, 3))
                layer.weight.grad.zero_()
            sensitivities.append(sensitivity)
        elif isinstance(layer, nn.Linear):
            sensitivity = torch.zeros(layer.out_features, layer.in_features)
            for j in range(layer.out_features):
                output = layer(inputs)
                loss = -output[:, j].sum()
                loss.backward(retain_graph=True)
                sensitivity[j] = layer.weight.grad.abs().mean(dim=1)
                layer.weight.grad.zero_()
            sensitivities.append(sensitivity)
    return sensitivities

# Define a function to compute the gradient receptive field of each neuron in a layer
def compute_gradient_receptive_field(model, inputs):
    gradient_rf = []
    for i in range(len(model)):
        layer = model[i]
        if isinstance(layer, nn.Conv2d):
            activation = inputs
            for j in range(i):
                activation = model[j](activation)
            activation.requires_grad_()
            output = layer(activation)
            loss = output.sum()
            loss.backward()
            gradient = activation.grad.abs()
            gradient_rf.append(gradient.mean(dim=(0, 2, 3)))
        elif isinstance(layer, nn.Linear):
            gradient_rf.append(None)
    return gradient_rf

# Instantiate the model and compute sensitivity and gradient receptive field for each layer
model = MyCNN()
inputs = torch.randn(1, 3, 32, 32)
sensitivities = compute_sensitivity(model, inputs)
gradient_rf = compute_gradient_receptive_field(model, inputs)

# Print the sensitivity and gradient receptive field for each layer
for i in range(len(model)):
    layer = model[i]
    print(f"Layer {i}: {layer}")
    if isinstance(layer, nn.Conv2d):
        print(f"Sensitivity: {sensitivities[i]}")
        print(f"Gradient

              
              
              
              
