import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.nets.student import *
from Net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

BatchNorm2d = SynchronizedBatchNorm2d


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabV3(nn.Module):
    def __init__(self, nclass, in_channels):
        print("Constructing DeepLabv3+ model...")
        print("Backbone: Resnet-101")
        print("Number of classes: {}".format(nclass))
        print("Output stride: {}".format(16))
        print("Number of Input Channels: {}".format(3))
        super(DeepLabV3, self).__init__()

        # Atrous Conv
        self.resnet18 = resnet18(in_channels)

        # ASPP
        dilations = [1, 6, 12, 18]

        self.aspp1 = ASPP_module(512, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(512, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(512, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(512, 256, dilation=dilations[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, nclass, kernel_size=1, stride=1))
        # if freeze_bn:
        #     self._freeze_bn()

    def forward(self, input):
        y1, y2, y3 = self.resnet18(input)
        #print(y1.size(), y2.size(), y3.size())
        x1 = self.aspp1(y3)
        x2 = self.aspp2(y3)
        x3 = self.aspp3(y3)
        x4 = self.aspp4(y3)
        x5 = self.global_avg_pool(y3)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        outputs = []

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(y1)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        outputs.append(x)

        return outputs

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabV3(nclass=2, in_channels=1)
    model.eval()
    image = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = model.forward(image)
    print(output[0].shape)
