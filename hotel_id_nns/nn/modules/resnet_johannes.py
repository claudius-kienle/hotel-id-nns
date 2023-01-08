import torch
import torch.nn as nn

class BigResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=None, stride=1):
        super().__init__()
        self.expension = 4

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expension, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expension)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

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

        # use down_sample layer if you need to change the shape
        if self.down_sample is not None:
            identity = self.down_sample(identity)

        x = x + identity
        x = self.relu(x)
        return x

class SmallResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def forward(self, x):

        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # use down_sample layer if you need to change the shape
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        x = x + identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, layers, num_classes, image_channels=3, use_big_resnet_blocks=True):
        super().__init__()
        self.expension = 4
        self.use_big_resnet_blocks = use_big_resnet_blocks
        self.in_channels = 64

        self.conv1 = nn.Sequential(nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2_x = self._create_conv_layer(layers[0], out_channels=64, stride=1)
        self.conv3_x = self._create_conv_layer(layers[1], out_channels=128, stride=2)
        self.conv4_x = self._create_conv_layer(layers[2], out_channels=256, stride=2)
        self.conv5_x = self._create_conv_layer(layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if use_big_resnet_blocks:
            self.fc = nn.Linear(512 * self.expension, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("forward1")
        x = self.conv1(x)
        print("forward2")
        x = self.conv2_x(x)
        print("forward3")
        x = self.conv3_x(x)
        print("forward4")
        x = self.conv4_x(x)
        print("forward5")
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


    # method to create conv2, conv3, conv4 and conv5
    # example for resnet 50 (big resnet_blocks)
    # conv1: layer1: in:64, out 64
    #        layer2: in 64, out 64
    #        layer3: in 64, out 256
    #
    #        layer4: in 256, out 64
    #        layer5: in 64, out 64
    #        layer6: in 64, out 256
    #
    #        ....
    #
    # conv2  layer 1: in 256, out 128
    #       layer 2: in 123, out 128
    #       layer 3: in 128, out 512
    #
    #       layer 4: in 512, out 128
    def _create_conv_layer(self, num_resnet_block, out_channels, stride):
        layers = []
        down_sample = None

        if self.use_big_resnet_blocks:
            down_sample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                                        nn.BatchNorm2d(out_channels * 4))
            # first layer has adapted identity down_sample
            layers.append(BigResNetBlock(self.in_channels, out_channels, down_sample, stride))
            self.in_channels = out_channels * 4

            # remaining layers without identity down_sample
            for layer_index in range(num_resnet_block - 1):
                layers.append(BigResNetBlock(self.in_channels, out_channels))

        else:
            if stride !=1:
                down_sample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                                        nn.BatchNorm2d(out_channels))
                layers.append(SmallResNetBlock(self.in_channels, out_channels, down_sample, stride))
                self.in_channels = out_channels


            else:
                layers.append(SmallResNetBlock(self.in_channels, out_channels))

            # remaining layers without identity down_sample
            for layer_index in range(num_resnet_block - 1):
                layers.append(SmallResNetBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)



def ResNet50(num_classes, image_channels=3):
    return ResNet([3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)

def ResNet101(num_classes, image_channels=3):
    return ResNet([3, 4, 23, 3], image_channels=image_channels, num_classes=num_classes)

def ResNet152(num_classes, image_channels=3):
    return ResNet([3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes)

def ResNet18(num_classes, image_channels=3):
    return ResNet([2, 2, 2, 2], image_channels=image_channels, num_classes=num_classes, use_big_resnet_blocks=False)

def ResNet34(num_classes, image_channels=3):
    return ResNet([3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes, use_big_resnet_blocks=False)


if __name__ == "__main__":
    net = ResNet18(3, image_channels=3)
    #net = ResNet50(3, image_channels=3)
    tensor = torch.randn(10, 3, 128, 128)
    y = net(tensor).to('cuda')
    print(y)
