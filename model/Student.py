from model.nets.student.MobileNetV2 import MobileNetV2
from model.nets.student.ResNet import *
from model.nets.models import *

in_channels = [128, 256, 512]


class Student(nn.Module):
    def __init__(self, n_classes, in_channels_1, in_channels_2=5):
        super(Student, self).__init__()
        self.n_classes = n_classes

        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2

        self.backbone = resnet18(in_channels_1)

        self.s1 = RCB(in_channels[0])
        self.s2 = RCB(in_channels[1])
        self.s3 = RCB(in_channels[2])

        self.FC_Spatical_Channel = SCConv(64, 128)
        self.Edge_Refinement = Edge_Refinement(self.n_classes, self.in_channels_2)

    def forward(self, x):
        s1, s2, s3 = self.backbone(x)

        s1 = self.s1(s1)
        s2 = self.s2(s2)
        s3 = self.s3(s3)

        s3 = F.upsample_bilinear(s3, s1.size()[2:])

        s = self.FC_Spatical_Channel(s1, s3)

        # print(s_out.shape)

        s_out = self.Edge_Refinement(s, s2, x)

        outputs = []
        outputs.append(s_out)

        return outputs


if __name__ == '__main__':
        x = torch.randn(1, 1, 256, 256)  # 创建随机输入张量
        model = Student(2, 3,7 )
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
