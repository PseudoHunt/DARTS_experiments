import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define Block Option 1: Basic Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

# Define Block Option 2: Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

# Define the NAS Model with 10 Layers
class NASModel(nn.Module):
    def __init__(self, search_space):
        super(NASModel, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        
        for layer_idx in range(10):
            config = search_space[layer_idx]
            block_type = config['block_type']
            out_channels = config['out_channels']
            stride = config.get('stride', 1)
            
            if block_type == 'conv':
                kernel_size = config.get('kernel_size', 3)
                self.layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            elif block_type == 'residual':
                self.layers.append(ResidualBlock(in_channels, out_channels, stride))

            in_channels = out_channels

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define the Search Space for Each of the 10 Layers
search_space = [
    [{'block_type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'stride': 1},
     {'block_type': 'residual', 'out_channels': 16, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'stride': 2},
     {'block_type': 'residual', 'out_channels': 32, 'stride': 2}],
    [{'block_type': 'conv', 'out_channels': 64, 'kernel_size': 3, 'stride': 2},
     {'block_type': 'residual', 'out_channels': 64, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 64, 'kernel_size': 3, 'stride': 1},
     {'block_type': 'residual', 'out_channels': 64, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 128, 'kernel_size': 3, 'stride': 2},
     {'block_type': 'residual', 'out_channels': 128, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 128, 'kernel_size': 3, 'stride': 1},
     {'block_type': 'residual', 'out_channels': 128, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 256, 'kernel_size': 3, 'stride': 2},
     {'block_type': 'residual', 'out_channels': 256, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 256, 'kernel_size': 3, 'stride': 1},
     {'block_type': 'residual', 'out_channels': 256, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 512, 'kernel_size': 3, 'stride': 2},
     {'block_type': 'residual', 'out_channels': 512, 'stride': 1}],
    [{'block_type': 'conv', 'out_channels': 512, 'kernel_size': 3, 'stride': 1},
     {'block_type': 'residual', 'out_channels': 512, 'stride': 1}],
]

# Initialize the NAS Model
model = NASModel(search_space)

# Load CIFAR-10 Data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define the Training Process
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):  # Train for 10 epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(trainloader)}')

# Test the Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
