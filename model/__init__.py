from .linearclassifier import LC
from .cnn import CNN, vgg_11
from .resnet import *

MODEL_MAP = {
    'resnet34':resnet_34,
    'resnet18':resnet_18,
    'resnet50':resnet_50,
    'vgg11':vgg_11,
    'lc':LC,
    'cnn':CNN
}

def remove_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from the state dictionary keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict