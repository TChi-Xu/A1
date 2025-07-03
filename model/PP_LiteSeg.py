import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.nets.student import *
from Net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d



class UAFM(nn.Module):
    def __init__(self, x_ch, y_ch):
        super().__init__()

        self.conv_x = nn.Sequential(nn.Conv2d(x_ch, x_ch, kernel_size=1),
                                    nn.BatchNorm2d(x_ch),
                                    nn.ReLU(inplace=True))
        self.conv_y = nn.Sequential(
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(y_ch, x_ch, kernel_size=1),
                                    nn.BatchNorm2d(x_ch),
                                    nn.ReLU(inplace=True)
        )

        self.conv_atten = nn.Sequential(nn.Conv2d(x_ch, x_ch//2, kernel_size=1),
                                        nn.BatchNorm2d(x_ch//2),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(x_ch//2, x_ch, kernel_size=1),)
        self.act = nn.Sigmoid()

        self.conv_out = nn.Sequential(nn.Conv2d(x_ch, x_ch, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(x_ch),
                                      nn.ReLU(inplace=True)
                                      )


    def forward(self, x, y):
        """
            Args:
                 x (Tensor): The low level feature.
                 y (Tensor): The high level feature.
        """
        x = self.conv_x(x)
        y = self.conv_y(y)
        out = x+y

        atten= F.avg_pool2d(out, out.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.act(atten)
        out = x*atten + y*(1-atten)
        out = self.conv_out(out)
        return out


class SPPM(nn.Module):
    def __init__(self):
        super(SPPM,self).__init__()
        self.global_avg_pool1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(512, 512, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(512),
                                             nn.ReLU(),
                                             )
        self.global_avg_pool2 = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)),
                                              nn.Conv2d(512, 512, 1, stride=1, bias=False),
                                              nn.BatchNorm2d(512),
                                              nn.ReLU()
                                              )
        self.global_avg_pool3 = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)),
                                              nn.Conv2d(512, 512, 1, stride=1, bias=False),
                                              nn.BatchNorm2d(512),
                                              nn.ReLU()
                                              )
        self.conv_out = nn.Sequential(nn.Conv2d(3*512, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self, x):
        global_avg_pool1 = self.global_avg_pool1(x)
        global_avg_pool2 = self.global_avg_pool2(x)
        global_avg_pool3 = self.global_avg_pool3(x)

        global_avg_pool1 = F.upsample_bilinear(global_avg_pool1, x.size()[2:])
        global_avg_pool2 = F.upsample_bilinear(global_avg_pool2, x.size()[2:])
        global_avg_pool3 = F.upsample_bilinear(global_avg_pool3, x.size()[2:])


        out = torch.cat((global_avg_pool1, global_avg_pool2, global_avg_pool3),1)

        out = self.conv_out(out)
        return out



class PP_LiteSeg(nn.Module):
    def __init__(self, nclass, in_channels):
        super(PP_LiteSeg, self).__init__()
        self.backbone = resnet18(in_channels)
        self.SPPM = SPPM()
        self.UAFM1 = UAFM(128, 256)
        self.UAFM2 = UAFM(256, 512)

        self.resize = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, nclass, kernel_size=1),
        )


    def forward(self,x):
        x1, x2, x3 = self.backbone(x)

        x3 = self.SPPM(x3)
        x2 = self.UAFM2(x2, x3)
        x1 = self.UAFM1(x1, x2)

        out = self.resize(x1)

        outputs = []

        outputs.append(out)

        return outputs


if __name__ == "__main__":
    model = PP_LiteSeg(nclass=2, in_channels=3)
    model.eval()
    image = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model.forward(image)
    print(output[0].shape)