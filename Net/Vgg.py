from torch import nn
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _load_weights(model, name):
    pre_trained_dict = torch.load('weights/train_weights/FCN{}_{}_potsdam_best_model.pth'.format(name[0], name[1]), map_location='cpu')
    model_dict = model.state_dict()
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    model_dict.update(pre_trained_dict)
    model.load_state_dict(model_dict)
    print('finish load weight')
    return model


def _vgg(pre_trained, cfg, batch_norm, name):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))
    if pre_trained:
        model = _load_weights(model, name)
    return model


def vgg11(pre_trained, name):
    """VGG 11-layer model (configuration "A")
    """
    return _vgg(pre_trained, 'A', False, name)


def vgg11_bn(pre_trained, name):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    return _vgg(pre_trained, 'A', True, name)


def vgg13(pre_trained, name):
    """VGG 13-layer model (configuration "B")
    """
    return _vgg(pre_trained, 'B', False, name)


def vgg13_bn(pre_trained, name):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    return _vgg(pre_trained, 'B', True, name)


def vgg16(pre_trained, name):
    """VGG 16-layer model (configuration "D")

    """
    return _vgg(pre_trained, 'D', False,name)


def vgg16_bn(pre_trained, name):
    """VGG 16-layer model (configuration "D") with batch normalization

    """
    return _vgg(pre_trained, 'D', True, name)


def vgg19(pre_trained, name):
    """VGG 19-layer model (configuration "E")
    """
    return _vgg(pre_trained, 'E', False, name)


def vgg19_bn(pre_trained, name):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    return _vgg(pre_trained, 'E', True, name)
