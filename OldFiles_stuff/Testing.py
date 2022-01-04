# loading packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

#matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.parameter import Parameter
from torchvision.datasets import MNIST
from torch import Tensor


# Importing the MNIST dataset
mnist_trainset = MNIST("./temp/", train=True, download=True)
mnist_testset = MNIST("./temp/", train=False, download=True)

# Only taking a subset of the data
x_train = mnist_trainset.data[:1000].view(-1, 784).float()
targets_train = mnist_trainset.targets[:1000]
x_test = mnist_testset.data[:500].view(-1, 784).float()
targets_test = mnist_testset.targets[:500]


'''
print("Information on dataset")
print("x_train", x_train.shape)
print("targets_train", targets_train.shape)
print("x_test", x_test.shape)
print("targets_test", targets_test.shape)
'''
#pytorch source code ResNet

# Har fjernet en del af de ting hvor de explicit skriver at ting = 1 når den ting = 1 per default
# Har fjernet nogle if's og gjort noget mere explicit, bland andet norm_layer har jeg bare gjort til batchnorm

# A basic block of the ResNet.
# Convolutions are also defined in here

def conv3x3(in_features: int, out_features: int):
    # 3x3 convolution with padding
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=False)

def conv1x1(in_features: int, out_features: int):
    # 1x1 convolution no padding
    # in_featues
    return nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1): 
        super(block, self).__init__()
        self.expansion = 4 # Number of blocks after a channel is always 4 times higher than when it entered; ref paper
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels) # normalize the batches, such that our output data don't variate too much 
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample # identity_downsample = convlayer, which we might need if we change the input sizes or number of channels
        
    def forward(self, x):
        identity = x

        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module): # [3,4,6,3]: how many times the blocks are used in each layer (4 layers)
    def __init__(self, block, layers, image_channels, num_classes): # image_channels= 3(RGB), 1(MNIST) etc. num_classes = how many classes we want to find(3,6,8 MNIST pictures) 
        super(ResNet, self).__init__()
        # Initialize modules
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3) # initial layer, haven't done anything of yet
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
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

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels,
                                                            kernel_size=1, stride=stride),nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = 64

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channels = 3, num_classes = 10000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def test():
    net = ResNet50()
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y.size())


test()

