import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    一个输入：来自于较深的特征层，
    两个输出：一个权重分布的图，一个是1-权重分布的图
    """

    def __init__(self, channels):
        super(Attention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.attention_layer(x)
        ones = torch.ones(x.size()).cuda()
        y = ones - x
        return x, y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Branch_Attention(nn.Module):
    """
    一个输入：来自于较深的特征层，
    两个输出：一个权重分布的图，一个是1-权重分布的图
    """

    def __init__(self, in_channels, out_channels):
        super(Branch_Attention, self).__init__()

        self.Semantic_Head1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )
        self.Semantic_Head2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )

        self.attention_layer1 = Attention(out_channels)
        self.attention_layer2 = Attention(out_channels)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, y):

        seg1 = self.Semantic_Head1(x)

        seg2 = self.Semantic_Head2(y)

        x_a, x_b = self.attention_layer1(x)

        out1 = seg1 * x_a
        out1 = F.interpolate(out1, y.size()[2:], mode='bilinear', align_corners=True)

        x_b = F.interpolate(x_b, y.size()[2:], mode='bilinear', align_corners=True)
        out2 = seg2 * x_b

        out = out1 + out2

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchsummary import summary
    from torchvision import models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # vgg = models.vgg19().to(device)
    model = Branch_Attention(64).to(device)
    input_data = torch.randn(1, 3, 512, 512)
    x, y = model.attention_layer(input_data)
    x = x.squeeze(0)[0]
    y = y.squeeze(0)[0]
    print(x)
    print(y)
    # print(model.attention_layer)
    # model = Attention(64).to(device)
    # vgg = Attention(64).to(device)

    # summary(model, [(3, 512, 512), (3, 512, 512)])
