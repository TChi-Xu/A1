
import torch.nn as nn
import torch.nn.functional as F

from Nets.LuNet import lu_net


class HH(nn.Module):
    def __init__(self, nclass, backbone='lu_net', aux=False, norm_layer=nn.BatchNorm2d):
        super(HH, self).__init__()
        self.aux = aux
        if backbone == 'lu_net':
            self.lu_net = lu_net()
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        net = list(self.lu_net.children())
        self.layer1_1 = nn.Sequential(*net[:2])
        # print(self.layer1.modules())
        self.layer1_2 = nn.Sequential(*net[2:4])
        self.attention1 = self.lu_net.Branch_Attention_512

        self.layer2_1 = nn.Sequential(*net[5])
        self.layer2_2 = nn.Sequential(*net[6:8])
        self.attention2 = self.lu_net.Branch_Attention_256

        self.layer3_1 = nn.Sequential(*net[9])
        self.layer3_2 = nn.Sequential(*net[10:12])
        self.attention3 = self.lu_net.Branch_Attention_128

        self.last_layer1 = self.lu_net.last_layer1
        self.last_layer2 = self.lu_net.last_layer2
        self.last_layer3 = self.lu_net.last_layer3

        self.head = _FCNHead(256, nclass, norm_layer)

        self.score_pool3 = nn.Conv2d(256, nclass, 1)
        self.score_pool4 = nn.Conv2d(256, nclass, 1)

    def forward(self, x):
        input1 = F.interpolate(x, scale_factor=2, mode="bilinear")
        input2 = x
        input3 = F.interpolate(x, scale_factor=0.5)
        input4 = F.interpolate(x, scale_factor=0.25)

        layer1_1 = self.layer1_1(input1)
        layer1_2 = self.layer1_2(input2)
        attention1 = self.attention1(layer1_2, layer1_1)

        layer2_1 = self.layer2_1(attention1)
        layer2_2 = self.layer2_2(input3)
        attention2 = self.attention2(layer2_2, layer2_1)

        layer3_1 = self.layer3_1(attention2)
        layer3_2 = self.layer3_2(input4)
        attention3 = self.attention3(layer3_2, layer3_1)

        last_layer1 = self.last_layer1(attention3)
        print(last_layer1.shape)
        last_layer2 = self.last_layer1(last_layer1)
        print(last_layer2.shape)
        last_layer3 = self.last_layer1(last_layer2)
        print(last_layer3.shape)

        outputs = []
        score_fr = self.head(last_layer3)

        score_pool4 = self.score_pool4(last_layer2)
        score_pool3 = self.score_pool3(last_layer1)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)
        outputs.append(out)

        return tuple(outputs)



class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    from torchsummary import summary
    from torchvision import models
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HH(6).to(device)
    # model = Branch_Attention(64).to(device)
    # input_data = torch.randn(1, 3, 512, 512)
    # x, y = model.attention_layer(input_data)
    # x = x.squeeze(0)[0]
    # y = y.squeeze(0)[0]
    # print(x)
    # print(y)
    # print(model.attention_layer)
    # model = Attention(64).to(device)
    # vgg = Attention(64).to(device)

    summary(model, (3, 512, 512))
