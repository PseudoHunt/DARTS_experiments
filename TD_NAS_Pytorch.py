import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorly.decomposition import partial_tucker

# Define the larger convolutional neural network
class LargeCNN(nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to apply Tucker decomposition to a convolutional layer
def tucker_decomposition_conv_layer(conv_layer, rank):
    weight_tensor = conv_layer.weight.data.clone().detach().cpu().numpy()
    core, [factors] = partial_tucker(weight_tensor, modes=[0, 1, 2, 3], rank=rank)
    return core, factors

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Create and train the original model
original_model = LargeCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(original_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = original_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

# Specify the convolutional layers to compress
conv_layers_to_compress = [original_model.conv2, original_model.conv4, original_model.conv6]

# Apply Tucker decomposition to compress convolutional layers
for conv_layer in conv_layers_to_compress:
    rank = [conv_layer.weight.size(0) // 2, 3, 3, conv_layer.weight.size(3) // 2]
    core, factors = tucker_decomposition_conv_layer(conv_layer, rank)
    conv_layer.weight.data = torch.Tensor(core)
    # Additional code if biases are present:
    # conv_layer.bias.data = torch.Tensor(factors[0].sum((1, 2, 3)))

# Create a new model with compressed layers
compressed_model = LargeCNN()
compressed_model.conv1 = original_model.conv1  # Copy unchanged layers

# Evaluate both models
original_model.eval()
compressed_model.eval()

correct_original, correct_compressed = 0, 0
total = 0

with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs_original = original_model(images)
        outputs_compressed = compressed_model(images)

        _, predicted_original = torch.max(outputs_original.data, 1)
        _, predicted_compressed = torch.max(outputs_compressed.data, 1)

        total += labels.size(0)
        correct_original += (predicted_original == labels).sum().item()
        correct_compressed += (predicted_compressed == labels).sum().item()

print(f'Accuracy of the original model on the 10000 test images: {100 * correct_original / total}%')
print(f'Accuracy of the compressed model on the 10000 test images: {100 * correct_compressed / total}%')
