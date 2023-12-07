import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet34, self).__init__()
        self.in_planes = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 1
        self.layer1_0 = block(self.in_planes, 64, stride=1)
        self.layer1_1 = block(64, 64, stride=1)
        self.layer1_2 = block(64, 64, stride=1)

        # Layer 2
        self.layer2_0 = block(64, 128, stride=2)
        self.layer2_1 = block(128, 128, stride=1)
        self.layer2_2 = block(128, 128, stride=1)
        self.layer2_3 = block(128, 128, stride=1)

        # Layer 3
        self.layer3_0 = block(128, 256, stride=2)
        self.layer3_1 = block(256, 256, stride=1)
        self.layer3_2 = block(256, 256, stride=1)
        self.layer3_3 = block(256, 256, stride=1)
        self.layer3_4 = block(256, 256, stride=1)
        self.layer3_5 = block(256, 256, stride=1)

        # Layer 4
        self.layer4_0 = block(256, 512, stride=2)
        self.layer4_1 = block(512, 512, stride=1)
        self.layer4_2 = block(512, 512, stride=1)

        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        # Forward pass through each residual block in the layers
        out = self.layer1_0(out)
        out = self.layer1_1(out)
        out = self.layer1_2(out)

        out = self.layer2_0(out)
        out = self.layer2_1(out)
        out = self.layer2_2(out)
        out = self.layer2_3(out)

        out = self.layer3_0(out)
        out = self.layer3_1(out)
        out = self.layer3_2(out)
        out = self.layer3_3(out)
        out = self.layer3_4(out)
        out = self.layer3_5(out)

        out = self.layer4_0(out)
        out = self.layer4_1(out)
        out = self.layer4_2(out)

        # Global average pooling
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        # Fully connected layer
        out = self.linear(out)

        return out

def ResNet34():
    return ResNet34(BasicBlock, [3, 4, 6, 3])

# Create an instance of ResNet-34
net = ResNet34()

# Test with a random input
random_input = torch.randn(1, 3, 32, 32)
output = net(random_input)
print(f'Output: {output.shape}')
