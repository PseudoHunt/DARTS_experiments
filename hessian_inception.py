import torch
import torch.nn as nn
import torch.optim as optim

# Define the Inception block
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv5 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = nn.ReLU()(self.conv1(x))
        x2 = nn.ReLU()(self.conv3(nn.ReLU()(self.conv2(x))))
        x3 = nn.ReLU()(self.conv5(nn.ReLU()(self.conv4(self.maxpool(x)))))
        return torch.cat([x1, x2, x3], dim=1)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.inception1 = InceptionBlock(32, 256)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inception2 = InceptionBlock(256, 512)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 512, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.inception1(x)
        x = self.maxpool1(x)
        x = self.inception2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 7 * 7 * 512)
        x = nn.ReLU()(self.fc1(x))
        x = nn.Dropout(p=0.5)(x)
        x = self.fc2(x)
        return x

# Create a toy dataset
X = torch.randn(10, 1, 28, 28)
y = torch.randint(0, 10, (10,))

# Initialize the model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model for a few epochs
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()

    # Compute the Hessian matrix for the Inception block
    hessian = torch.autograd.functional.hessian(lambda x: model.inception1(x).mean(), model(X))[0]

    # Compute the sensitivity of each branch
    for i in range(3):
        v = torch.zeros(hessian.shape[0], 1, 1)
        v[i, 0, 0] = 1
        branch_s
# Compute the sensitivity of each branch
for i in range(3):
    v = torch.zeros(hessian.shape[0], 1, 1, 1)
    v[:, i::3, :, :] = 1
    sensitivity = torch.abs(torch.matmul(torch.matmul(hessian, v), v.transpose(1, 0)))
    print(f"Sensitivity of branch {i+1}: {sensitivity.item()}")
