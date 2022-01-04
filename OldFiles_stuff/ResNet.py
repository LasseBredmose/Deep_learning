# Creating the network
import torch.nn as nn

class baseblock(nn.Module):
    expansion = 1
    basic = True
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(baseblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample # identity_downsample = convlayer, which we might need if we change the input sizes or number of channels

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class block(nn.Module):
    expansion = 4 # Number of blocks after a channel is always 4 times higher than when it entered; ref paper
    basic = False
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # normalize the batches, such that our output data don't variate too much 
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
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
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False) # initial layer, haven't done anything of yet
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512* block.expansion, num_classes)

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
        if block.basic:
            stride =1

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels*block.expansion,
                                                            kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(out_channels*block.expansion))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        
        self.in_channels = out_channels * block.expansion

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channels = 1, num_classes = 10):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet18(img_channels = 1, num_classes = 10):
    return ResNet(baseblock, [2,2,2,2], img_channels, num_classes)
    
def ResNetX(img_channels = 1, num_classes = 10, layers = [2,2,2,2]):
    return ResNet(baseblock, layers, img_channels, num_classes )

'''

def test():
    net = ResNet50()
    x = torch.randn(2, 1, 100, 100)  # 4 dim, 2 pictures with 3 channels af 224 pixels in each. 
    y = net(x)
    print(y.size())

#test()
'''