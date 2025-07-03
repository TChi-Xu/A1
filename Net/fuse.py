import torch
from torch import nn
import torch.nn.functional as F


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
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )
        self.Semantic_Head2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )
        self.Semantic_Head3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, y, z):
        x = self.Semantic_Head1(x)
        x = self.down(x)
        y = self.Semantic_Head2(y)
        z = self.Semantic_Head3(z)
        z = F.interpolate(z, y.size()[2:], mode='bilinear', align_corners=True)
        out = x + y + z

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
