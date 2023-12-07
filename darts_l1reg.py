import torch
import torch.nn as nn
import torch.optim as optim

# Define the search space (operations)
OPS = {
    'conv3x3': nn.Conv2d(1, 1, 3, 1, 1),
    'conv5x5': nn.Conv2d(1, 1, 5, 1, 2),
    'pool3x3': nn.MaxPool2d(3, 1, 1),
    'pool5x5': nn.MaxPool2d(5, 1, 2)
}

# Define the operations for the cell
class Cell(nn.Module):
    def __init__(self, op_names, prev, curr):
        super(Cell, self).__init__()
        self.ops = nn.ModuleList([OPS[op] for op in op_names])
        self.weights = nn.Parameter(torch.randn(len(op_names)))

    def forward(self, x_prev, x_curr):
        return sum(w * op(x_prev if i < len(x_prev) else x_curr) for i, (w, op) in enumerate(zip(self.weights, self.ops)))

# Define the DARTS network
class DARTSNetwork(nn.Module):
    def __init__(self, num_cells):
        super(DARTSNetwork, self).__init__()
        self.cells = nn.ModuleList([Cell(['conv3x3', 'conv5x5', 'pool3x3', 'pool5x5'], [0, 1], 2) for _ in range(num_cells)])

    def forward(self, x):
        states = [x]
        for cell in self.cells:
            states.append(cell(states[:-2], states[-1]))
        return sum(states[-2:])

    def arch_parameters(self):
        # Return architecture parameters for optimization
        return [param for name, param in self.named_parameters() if 'weights' in name]

# Instantiate the DARTS network
darts_net = DARTSNetwork(num_cells=5)

# Define a dummy input
input_data = torch.randn(1, 1, 32, 32)  # Replace with your input size

# Set up optimizer for architecture parameters with L1 regularization
arch_optimizer = optim.SGD(darts_net.arch_parameters(), lr=0.1)

# Set up optimizer for model parameters
model_optimizer = optim.SGD(darts_net.parameters(), lr=0.01)

# L1 regularization strength
l1_lambda = 0.001  # Adjust the strength as needed

# Number of optimization steps
num_steps = 100

# Training loop
for step in range(num_steps):
    # Forward pass
    output = darts_net(input_data)

    # Compute loss with L1 regularization
    loss = torch.sum(output)
    
    # Add L1 regularization to the loss
    arch_params = darts_net.arch_parameters()
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in arch_params:
        l1_reg = l1_reg + torch.norm(param, p=1)
    loss = loss + l1_lambda * l1_reg

    # Clear previous gradients
    arch_optimizer.zero_grad()
    model_optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update architecture parameters
    arch_optimizer.step()

    # Update model parameters
    model_optimizer.step()

# Print the optimized architecture parameters
for name, param in darts_net.named_parameters():
    if 'weights' in name:
        print(name, param.data)
