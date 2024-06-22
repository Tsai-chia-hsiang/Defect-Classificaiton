import json
from typing import Any
import os

def read_json(f:os.PathLike)->Any:
    ret = None
    with open(f, "r") as fp:
        ret = json.load(fp)
    return ret

def write_json(O, f:os.PathLike)->Any:
    ret = None
    with open(f, "w+") as fp:
        json.dump(O, fp=fp, indent=4, ensure_ascii=False)

