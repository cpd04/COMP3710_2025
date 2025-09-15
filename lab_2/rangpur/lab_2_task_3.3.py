# helpimport torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def task_3():
    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU may be slow.")

    # Hyper-parameters
    num_epochs = 35
    learning_rate = 0.1
    batch_size = 128
    depth = 64

    # Setup and normalise the data by pre-processing - easier than Tensorflow
    transform_train = transforms.Compose([
        transforms.ToTensor(), # Convert images to PyTorch tensors
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalise to mean and std of CIFAR-10, pre-computed for each channel
        transforms.RandomHorizontalFlip(), # Data augmentation to help with overfitting
        transforms.RandomCrop(32, padding=4, padding_mode='reflect') # Data augmentation to help with overfitting
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6) #num_workers depends on your CPU

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6) 

    # In PyTorch, we always use custom 'Modules' as classes for better robustness 
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1): # How many inputs and what depth
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes) # Used to normalise everything in between to ensure no crazy values
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        def forward(self, x): # Forward pass, what happens when you pass data through
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        
        # Now make resnet 18 - outside of scope but combination of blocks in different ways
    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4) # Global average pooling to actually reduce size before flatten and dense linear out
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])

    model = ResNet18().to(device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Model info
    print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
    print(model)

    criterion = nn.CrossEntropyLoss() # Good for multi-class classification, expects class labels not one-hot
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) # Weight decay is L2 regularization, SGD is really good for speed

    # Piecewise Linear Schedule - allows you to get 94% accuracy in 1/10th of the time
    total_step = len(trainloader)
    sched_linear_1 = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=learning_rate, step_size_up=15, step_size_down=15, mode='triangular') 
    sched_linear_3 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/learning_rate, end_factor=0.005/5) 
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30]) 

    # Train the model
    model.train()
    print("> Training")
    start = time.time()
    for epoch in range(num_epochs):
        
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        scheduler.step() # Update learning rate

    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    # Test the model
    print("> Testing")
    start = time.time()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Test Accuracy: {} %'.format(100 * correct / total))

    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

if __name__ == "__main__":
    # Run Function
    task_3()
    