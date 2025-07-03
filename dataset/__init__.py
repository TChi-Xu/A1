"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .HISEA import HISEA
from .ETCI_2021 import ETCI_2021
from .GF_FloodNet import GF_FloodNet
# from .voc2012 import Voc2012

datasets = {
    'hisea': HISEA,
    'etci_2021': ETCI_2021,
    'gf_floodnet': GF_FloodNet
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
