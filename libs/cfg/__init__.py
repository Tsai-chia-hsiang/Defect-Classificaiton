from .default import _C
import os
from yacs.config import CfgNode as CN

def load_config(yaml_file: os.PathLike) -> CN:
    setting = None    
    with open(yaml_file, "r") as s:
        setting = CN.load_cfg(s)
    setting.model.depth = str(setting.model.depth)
    
    _C.merge_from_other_cfg(setting) 
    return _C