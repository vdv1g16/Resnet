import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import sampler
from tensorboardX import SummaryWriter
import time
repo_path = os.path.dirname(os.path.abspath(__file__))
print("repo_path is", repo_path)
sys.path.append(repo_path)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
learning_rate = 0.001
NUM_TRAIN = 45000
# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='/home/vdv1g16/Desktop/Python_learning/ResNet/data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=200, 
                                           sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

val_dataset = torchvision.datasets.CIFAR10(root='/home/vdv1g16/Desktop/Python_learning/ResNet/data/', 
                                            train=True, 
                                            transform=transform, 
                                            download=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                        batch_size=200, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

test_dataset = torchvision.datasets.CIFAR10(root='/home/vdv1g16/Desktop/Python_learning/ResNet/data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

writer = SummaryWriter("data/tensorboard/")    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0 
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, preds = torch.max(scores.data, 1)
            num_correct += (preds == y).sum().item()
            
            num_samples += preds.size(0)
        print("num_correct is", num_correct)
        acc = float(num_correct) / num_samples * 100
        print("Got %d / %d  correct (%.2f)" % (num_correct, num_samples, acc))
    return acc

    
# Train the model
total_step = len(train_loader)
print("total_step is", total_step)
curr_lr = learning_rate
start_time = time.time()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        step_num = i + epoch*total_step
        print("Epoch {} iter {}".format(epoch, step_num))
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), i + epoch*total_step)
        if (i + epoch*total_step) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            acc = check_accuracy(val_loader, model)
            writer.add_scalar("eval_accuracy", acc, step_num)
            print(" Accuracy at step %d is (%.2f)" %(step_num, acc))
            print(" Time at step {} is {} s".format(step_num, time.time() - start_time))


    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
acc = check_accuracy(test_loader, model)
writer.add_scalar("test_accuracy", acc, 121)

torch.save(model.state_dict(), 'data/saved_models/resnet.ckpt')