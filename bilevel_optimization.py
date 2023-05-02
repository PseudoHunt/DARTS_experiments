import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
C = 16  # number of channels
layers = 8  # number of layers
num_classes = 10  # number of classes
epochs = 50  # number of epochs
batch_size = 128  # batch size
learning_rate = 0.025  # learning rate
momentum = 0.9  # momentum
weight_decay = 3e-4  # weight decay

# Define the transformations to use for the dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load the CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Create the model and optimizer
model = Network(C=C, num_classes=num_classes, layers=layers, criterion=criterion).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Train the model
for epoch in range(epochs):
    # Set the model to train mode
    model.train()

    # Initialize the total loss and correct predictions for this epoch
    total_loss = 0.0
    correct = 0

    # Iterate over the batches in the train_loader
    for i, (inputs, targets) in enumerate(train_loader):
        # Move the inputs and targets to the device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Update the cell weights by fixing the layer weights
        model.fix_layer_weights()
        optimizer_cell = optim.Adam(model.cell_parameters(), lr=learning_rate)
        optimizer_cell.zero_grad()
        outputs = model(inputs, None)
        loss_cell = criterion(outputs, targets)
        loss_cell.backward()
        optimizer_cell.step()

        # Update the layer weights by fixing the cell weights
        model.fix_cell_weights()
        optimizer_layer = optim.SGD(model.layer_parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_layer.zero_grad()
        outputs = model(inputs, None)
        loss_layer = criterion(outputs, targets)
        loss_layer.backward()
        optimizer_layer.step()

        # Add the loss to the total loss
        total_loss += (loss_cell.item() + loss_layer.item())

        # Compute the number of correct predictions
        _, predicted = outputs.max(1)
import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion):
        super(Network, self).__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.criterion = criterion
        
        # Define the initial stem convolution layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        
        # Define the normal and reduction cells
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_out = C * 2
                reduction = True
            else:
                C_out = C
                reduction = False
            cell = Cell(C_in=C, C_out=C_out, reduction=reduction)
            self.cells.append(cell)
        
        # Define the classifier
        self.classifier = nn.Linear(C, num_classes)
        
        # Initialize the layer and cell weights
        self.initialize_weights()

        # Set the flag for which weights to fix
        self.fix_cell = False

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def fix_layer_weights(self):
        self.fix_cell = True

    def fix_cell_weights(self):
        self.fix_cell = False

    def cell_parameters(self):
        if self.fix_cell:
            return []
        else:
            return list(self.cells.parameters())

    def layer_parameters(self):
        if self.fix_cell:
            return list(self.stem.parameters()) + list(self.classifier.parameters())
        else:
            return list(self.stem.parameters()) + list(self.cells.parameters()) + list(self.classifier.parameters())

    def forward(self, x, genotype):
        # Perform the stem convolution
        x = self.stem(x)

        # Iterate over the cells and apply them to the feature maps
        s0 = s1 = x
        for i, cell in enumerate(self.cells):
            if genotype is None:
                weights = None
            else:
                weights = genotype[i]
            s0, s1 = s1, cell(s0
