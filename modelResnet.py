import torch
import torch.nn as nn
import torchvision.models as models
# Definisikan blok dasar ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

pretrained_resnet18 = models.resnet18(pretrained=True)

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        # Remove the last layer of pretrained ResNet18
        self.features = nn.Sequential(*list(pretrained_resnet18.children())[:-1])

        # Add a new layer
        self.additional_layer = nn.Linear(512, 512)  # Update the input size to match the output size of self.features
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.additional_layer(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
