import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    一个输入：来自于较深的特征层，
    两个输出：一个权重分布的图，一个是1-权重分布的图
    """

    def __init__(self, channels, threshold, nclass):
        super(Attention, self).__init__()
        self.threshold1 = threshold[0]
        self.threshold2 = threshold[1]
        self.nclass = nclass
        self.attention_layer1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, self.nclass, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.nclass),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nclass, self.nclass, 1, bias=False),
            nn.Sigmoid())
        self.attention_layer2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, self.nclass, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.nclass),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nclass, self.nclass, 1, bias=False),
            nn.Sigmoid())
        self.Semantic_Head1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, self.nclass, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.nclass),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nclass, self.nclass, 1, bias=False),
        )
        self.Semantic_Head2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, self.nclass, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.nclass),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nclass, self.nclass, 1, bias=False),
        )
        self.Semantic_Head3 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, self.nclass, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.nclass),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nclass, self.nclass, 1, bias=False),
        )

    def forward(self, x1, y1, z1):
        x2 = self.Semantic_Head1(x1)
        # x2 = x1 + x2
        y2 = self.Semantic_Head2(y1)
        # y2 = y1 + y2
        z2 = self.Semantic_Head3(z1)
        # z2 = z1 + z2

        a = self.attention_layer1(x1)
        x = ((a >= self.threshold1).float() * x2).cuda()

        b = self.attention_layer2(y1)
        y2 = ((b >= self.threshold2).float() * y2).cuda()

        z2 = F.interpolate(z2, y1.size()[2:], mode='bilinear', align_corners=True)
        z2 = ((b < self.threshold2).float() * z2).cuda()

        yz = y2 + z2
        yz = F.interpolate(yz, x1.size()[2:], mode='bilinear', align_corners=True)
        y = ((a < self.threshold1).float() * yz).cuda()

        out = x + y
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


class Branch_Attention(nn.Module):
    """
    一个输入：来自于较深的特征层，
    两个输出：一个权重分布的图，一个是1-权重分布的图
    """

    def __init__(self, channels):
        super(Branch_Attention, self).__init__()
        self.attention_layer = Attention(channels)
        self.Semantic_Head1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
        )
        self.Semantic_Head2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
        )

    def forward(self, x, y):
        x1 = self.Semantic_Head1(x)
        y1 = self.Semantic_Head2(y)
        y1 = F.interpolate(y1, x.size()[2:], mode='bilinear', align_corners=True)
        x2, y2 = self.attention_layer(x)
        x3 = x1 * x2
        y3 = y1 * y2
        out = x3 + y3
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
