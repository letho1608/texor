"""ResNet model implementation"""
from typing import List
from ...core import Tensor
from ..layers import Layer, Conv2d, BatchNorm2d, MaxPool2d, Linear, Sequential, AdaptiveAvgPool2d
from ..activations import ReLU
from ..advanced_layers import ResidualBlock

class ResNet(Layer):
    """ResNet implementation"""
    def __init__(self, num_layers: int, num_classes: int = 1000):
        super().__init__()
        if num_layers not in [18, 34, 50, 101, 152]:
            raise ValueError("Supported ResNet layers: 18, 34, 50, 101, 152")
            
        self.in_channels = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet configurations
        configs = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        
        self.layer1 = self._make_layer(64, configs[num_layers][0])
        self.layer2 = self._make_layer(128, configs[num_layers][1], stride=2)
        self.layer3 = self._make_layer(256, configs[num_layers][2], stride=2)
        self.layer4 = self._make_layer(512, configs[num_layers][3], stride=2)
        
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, num_classes)
        
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> Sequential:
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return Sequential(layers)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

def resnet18(num_classes: int = 1000) -> ResNet:
    """ResNet-18 model"""
    return ResNet(18, num_classes)

def resnet34(num_classes: int = 1000) -> ResNet:
    """ResNet-34 model"""
    return ResNet(34, num_classes)

def resnet50(num_classes: int = 1000) -> ResNet:
    """ResNet-50 model"""
    return ResNet(50, num_classes)

def resnet101(num_classes: int = 1000) -> ResNet:
    """ResNet-101 model"""
    return ResNet(101, num_classes)

def resnet152(num_classes: int = 1000) -> ResNet:
    """ResNet-152 model"""
    return ResNet(152, num_classes)