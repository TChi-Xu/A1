from torch import nn
import torch.nn.functional as F
from Net.Attention import Branch_Attention


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    inplanes :指的是输入的通道
    planes ：输出的通道数

    """
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LuNet(nn.Module):

    def __init__(self, block, layers, dilation=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(LuNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.input_channels = 64
        self.dilation = dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [2, 5, 9]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.top_layer = nn.Sequential(
            nn.Conv2d(3, self.input_channels, kernel_size=7, stride=2, padding=3,
                      bias=False),
            norm_layer(self.input_channels),
            nn.ReLU(inplace=True))  # 512x512x64
        # layer_1024x1024 没有下采样，输入通道：64， 输出通道128，无下采样：即stride=1，膨胀率：无 512x512x128
        self.layer_1024 = self._make_layer(block, 64, layers[0])
        # layer_512x512 没有下采样，输入通道：64， 输出通道128，无下采样：即stride=1，膨胀率：无 512x512x128
        self.top_512 = nn.Sequential(
            nn.Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            norm_layer(self.input_channels),
            nn.ReLU(inplace=True))  # 512x512x64
        # print('top_512')
        self.layer_512 = self._make_layer(block, 64, layers[0])
        # print('layer_512')
        self.Branch_Attention_512 = Branch_Attention(128)

        self.layer512_256 = self._make_layer(block, 128, layers[1], stride=2,
                                             dilate=replace_stride_with_dilation[0])

        self.top_256 = nn.Sequential(
            nn.Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            norm_layer(self.input_channels),
            nn.ReLU(inplace=True))  # 256x256x64
        # 256x256x128
        self.layer_256 = self._make_layer(block, 128, layers[0])
        self.Branch_Attention_256 = Branch_Attention(256)
        # 128x128x256
        self.layer256_128 = self._make_layer(block, 128, layers[1], stride=2,
                                             dilate=replace_stride_with_dilation[1])

        self.top_128 = nn.Sequential(
            nn.Conv2d(3, self.input_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            norm_layer(self.input_channels),
            nn.ReLU(inplace=True))  # 128x128x64
        # 128x128x256
        self.layer_128 = self._make_layer(block, 128, layers[0])
        self.Branch_Attention_128 = Branch_Attention(256)
        # 64x64x256
        self.last_layer1 = self._make_layer(block, 128, layers[0], stride=2,
                                            dilate=replace_stride_with_dilation[1])
        # 32x32x512
        self.last_layer2 = self._make_layer(block, 256, layers[0], stride=2,
                                            dilate=replace_stride_with_dilation[1])
        # 16x16x512
        self.last_layer3 = self._make_layer(block, 256, layers[0], stride=2,
                                            dilate=replace_stride_with_dilation[1])
        # self.layer3 = self._make_layer(block, 256, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, channels, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            # stride = 1
        if stride != 1 or self.input_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_channels, channels * block.expansion, stride),
                norm_layer(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.input_channels, channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.input_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.input_channels, channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        input1 = F.interpolate(x, scale_factor=2, mode="bilinear")
        input2 = x
        input3 = F.interpolate(x, scale_factor=0.5)
        input4 = F.interpolate(x, scale_factor=0.25)
        top_layer = self.top_layer(input1)
        layer_1024 = self.layer_1024(top_layer)
        # print('layer_1024')
        top_512 = self.top_512(input2)
        # print('top_512')
        layer_512 = self.layer_512(top_512)
        # print('layer_512')
        Branch_Attention_512 = self.Branch_Attention_512(layer_512, layer_1024)
        layer512_256 = self.layer512_256(Branch_Attention_512)  # 256x256x256
        top_256 = self.top_256(input3)  # 256x256x64
        layer_256 = self.layer_256(top_256)  # 256x256x256
        Branch_Attention_256 = self.Branch_Attention_256(layer_256, layer512_256)
        layer256_128 = self.layer256_128(Branch_Attention_256)
        top_128 = self.top_128(input4)  # 256x256x64
        layer_128 = self.layer_128(top_128)
        Branch_Attention_128 = self.Branch_Attention_128(layer_128, layer256_128)
        last_layer1 = self.last_layer1(Branch_Attention_128)
        last_layer2 = self.last_layer2(last_layer1)
        last_layer3 = self.last_layer3(last_layer2)
        return last_layer3


def _lu_net(block, layers):
    model = LuNet(block, layers)
    return model


def lu_net():
    """Constructs a ResNet-18 model.
    """
    return _lu_net(Bottleneck, [3, 4])


if __name__ == '__main__':
    # from torchsummary import summary
    # from torchvision import models
    import torch
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = LuNet(Bottleneck, [3, 4])
    # # model = Branch_Attention(64).to(device)
    # # input_data = torch.randn(1, 3, 512, 512)
    # # x, y = model.attention_layer(input_data)
    # # x = x.squeeze(0)[0]
    # # y = y.squeeze(0)[0]
    # # print(x)
    # # print(y)
    # # print(model.attention_layer)
    # # model = Attention(64).to(device)
    # # vgg = Attention(64).to(device)
    #
    # summary(model, (3, 512, 512))

    net = lu_net().to(device)
    # for i in net.modules():
    #     print('+' * 20)
    #     print(i)
    #     print('+' * 20)
    net = list(net.children())
    # print(len(net))
    # print(net[13])
    print(type(net))
    print(len(net))
    # print(*net[:2])
    print(net[13])
    # print(net)