# https://arxiv.org/pdf/1512.03385.pdf
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn
from hotel_id_nns.nn.modules.conv_layer import ConvLayer

from hotel_id_nns.nn.modules.global_avg_pool_2d import GlobalAvgPool2d


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_channel_list: List[Tuple[int, int]],
                 sample_down: bool):
        super().__init__()

        self.depth = len(kernel_channel_list)
        self.sample_down = sample_down

        c_in = in_channels
        for idx, (kernel_size, c_out) in enumerate(kernel_channel_list):
            stride = 2 if idx == 1 and sample_down else 1  # sample down at 3x3 convolution
            self.add_module(
                f"conv_{idx}",
                nn.Conv2d(kernel_size=kernel_size,
                          in_channels=c_in,
                          out_channels=c_out,
                          stride=stride,
                          padding=kernel_size // 2,
                          bias=False))

            self.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(num_features=c_out))

            self.add_module(f"relu_{idx}", nn.ReLU())

            c_in = c_out

        block_out_channels = c_in

        if sample_down or in_channels != block_out_channels:
            stride = 2 if sample_down else 1
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    kernel_size=1,
                    in_channels=in_channels,
                    out_channels=block_out_channels,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=block_out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        y = x
        for idx in range(self.depth):
            y = getattr(self, f"conv_{idx}")(y)
            y = getattr(self, f"batch_norm_{idx}")(y)

            if idx < self.depth - 1:
                y = getattr(self, f"relu_{idx}")(y)

        if self.downsample is not None:
            x = self.downsample(x)

        y = y + x

        y = getattr(self, f"relu_{self.depth - 1}")(y)

        return y



def build_conv2d(in_channels, out_channels, stride, kernel_size):
    return ConvLayer(in_channels=in_channels,
                     out_channels=out_channels,
                     stride=stride,
                     kernel_size=kernel_size,
                     padding=kernel_size // 2,
                     bias=False)


def build_sequential(args: List[dict]):
    modules = [cfg["builder"](**cfg["args"]) for cfg in args]
    return nn.Sequential(*modules)


def build_max_pool(kernel_size, stride):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)


def build_resnet_block(in_channels: int,
                       kernel_channel_list: List[Tuple[int, int]],
                       sample_down: bool = False):
    return ResNetBlock(in_channels, kernel_channel_list, sample_down)


def build_global_average_pooling():
    return nn.AdaptiveAvgPool2d((1, 1))


resnet18_cfg = dict(conv1=dict(builder=build_conv2d,
                               args=dict(in_channels=3, out_channels=64, stride=2, kernel_size=7)),
                    conv2_x=dict(builder=build_sequential,
                                 args=[
                                     dict(builder=build_max_pool,
                                          args=dict(kernel_size=3, stride=2)),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=64,
                                                    kernel_channel_list=[(3, 64), (3, 64)])),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=64,
                                                    kernel_channel_list=[(3, 64), (3, 64)])),
                                 ]),
                    conv3_x=dict(builder=build_sequential,
                                 args=[
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=64,
                                                    kernel_channel_list=[(3, 128), (3, 128)],
                                                    sample_down=True)),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=128,
                                                    kernel_channel_list=[(3, 128), (3, 128)])),
                                 ]),
                    conv4_x=dict(builder=build_sequential,
                                 args=[
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=128,
                                                    kernel_channel_list=[(3, 256), (3, 256)],
                                                    sample_down=True)),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=256,
                                                    kernel_channel_list=[(3, 256), (3, 256)])),
                                 ]),
                    conv5_x=dict(builder=build_sequential,
                                 args=[
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=256,
                                                    kernel_channel_list=[(3, 512), (3, 512)],
                                                    sample_down=True)),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=512,
                                                    kernel_channel_list=[(3, 512), (3, 512)])),
                                 ]),
                    global_average_pooling=dict(builder=build_global_average_pooling),
                    out_features=512)

_resnet50_conv3_1 = [
    dict(builder=build_resnet_block,
         args=dict(in_channels=256,
                   kernel_channel_list=[(1, 128), (3, 128), (1, 512)],
                   sample_down=True))
]
_resnet50_conv3_234 = [
    dict(builder=build_resnet_block,
         args=dict(in_channels=512,
                   kernel_channel_list=[(1, 128), (3, 128), (1, 512)],
                   sample_down=False)) for _ in range(3)
]
_resnet50_conv3_x = _resnet50_conv3_1 + _resnet50_conv3_234

_resnet50_conv4_1 = [
    dict(builder=build_resnet_block,
         args=dict(in_channels=512,
                   kernel_channel_list=[(1, 256), (3, 256), (1, 1024)],
                   sample_down=True))
]
_resnet50_conv4_23456 = [
    dict(builder=build_resnet_block,
         args=dict(in_channels=1024,
                   kernel_channel_list=[(1, 256), (3, 256), (1, 1024)],
                   sample_down=False)) for _ in range(5)
]
_resnet50_conv4_x = _resnet50_conv4_1 + _resnet50_conv4_23456

resnet50_cfg = dict(conv1=dict(builder=build_sequential,
                               args=[
                                   dict(builder=build_conv2d,
                                        args=dict(in_channels=3,
                                                  out_channels=64,
                                                  stride=2,
                                                  kernel_size=7)),
                                   dict(builder=build_max_pool, args=dict(kernel_size=3, stride=2)),
                               ]),
                    conv2_x=dict(builder=build_sequential,
                                 args=[
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=64,
                                                    kernel_channel_list=[(1, 64), (3, 64),
                                                                         (1, 256)])),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=256,
                                                    kernel_channel_list=[(1, 64), (3, 64),
                                                                         (1, 256)])),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=256,
                                                    kernel_channel_list=[(1, 64), (3, 64),
                                                                         (1, 256)])),
                                 ]),
                    conv3_x=dict(builder=build_sequential, args=_resnet50_conv3_x),
                    conv4_x=dict(builder=build_sequential, args=_resnet50_conv4_x),
                    conv5_x=dict(builder=build_sequential,
                                 args=[
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=1024,
                                                    kernel_channel_list=[(1, 512), (3, 512),
                                                                         (1, 2048)],
                                                    sample_down=True)),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=2048,
                                                    kernel_channel_list=[(1, 512), (3, 512),
                                                                         (1, 2048)])),
                                     dict(builder=build_resnet_block,
                                          args=dict(in_channels=2048,
                                                    kernel_channel_list=[(1, 512), (3, 512),
                                                                         (1, 2048)])),
                                 ]),
                    global_average_pooling=dict(builder=build_global_average_pooling),
                    out_features=2048)

_resnet101_conv4_2_to_23 = [
    dict(builder=build_resnet_block,
         args=dict(in_channels=1024,
                   kernel_channel_list=[(1, 256), (3, 256), (1, 1024)],
                   sample_down=False)) for _ in range(22)
]
_resnet101_conv4_x = _resnet50_conv4_1 + _resnet101_conv4_2_to_23

resnet101_cfg = deepcopy(resnet50_cfg)
resnet101_cfg["conv4_x"]["args"] = _resnet101_conv4_x


class ResNet(torch.nn.Module):
    def __init__(self, network_cfg: dict, out_features):
        # How to init weights?
        super().__init__()

        conv1_cfg = network_cfg["conv1"]
        conv2_x_cfg = network_cfg["conv2_x"]
        conv3_x_cfg = network_cfg["conv3_x"]
        conv4_x_cfg = network_cfg["conv4_x"]
        conv5_x_cfg = network_cfg["conv5_x"]

        self.conv1 = conv1_cfg["builder"](conv1_cfg["args"])
        self.conv2_x = conv2_x_cfg["builder"](conv2_x_cfg["args"])
        self.conv3_x = conv3_x_cfg["builder"](conv3_x_cfg["args"])
        self.conv4_x = conv4_x_cfg["builder"](conv4_x_cfg["args"])
        self.conv5_x = conv5_x_cfg["builder"](conv5_x_cfg["args"])
        self.global_average_pooling = network_cfg["global_average_pooling"]["builder"]()

        resnet_out = network_cfg["out_features"]

        self.fully_connected = nn.Linear(in_features=resnet_out, out_features=out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.global_average_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected(x)
        return x
    

if __name__ == "__main__":
    net = ResNet(resnet101_cfg, 1000)
    print(net)

    x = torch.zeros((2, 3, 224, 224))
    y = net(x)
    print(y.shape)
