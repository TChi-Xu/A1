from .Student import *
from .Teacher import *

from .DeepLabV3 import *
from .PSPNet import *
from .CCNet import *
from .WaveResUNet import *
from .LRNNet import *
from .PP_LiteSeg import *
from .RepViT import *
from .AFFormer import *
from .LightReSeg import *
from .GCtx_UNet import *


Get_Models = {
    'Student': Student,
    'Teacher': Teacher,
    'DeepLabV3': DeepLabV3,
    'PSPNet': PSPNet,
    'CCNet': CCNet,
    'WaveResUNet': WaveResUNet,
    'LRNNet': LRNNet,
    'PP_LiteSeg': PP_LiteSeg,
    'RepViT': repvit_m1_5,
    'AFFormer': Afformer_base,
    'LightReSeg': LightReSeg,
    'GCtx_UNet': GCtx_UNet,


}


def get_segmentation_Model(name, **data_kwargs):
    """Segmentation Datasets"""
    return Get_Models[name](**data_kwargs)
