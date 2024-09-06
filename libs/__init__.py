import json
from typing import Any
import os
import random
import numpy as np
import torch

def read_json(f:os.PathLike)->Any:
    ret = None
    with open(f, "r") as fp:
        ret = json.load(fp)
    return ret

def write_json(O, f:os.PathLike)->Any:
    with open(f, "w+") as fp:
        json.dump(O, fp=fp, indent=4, ensure_ascii=False)

def set_seed(seed=42):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    