import os
from logging import Logger
from typing import Any, Callable, Optional
from typing import Literal
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

G_normalizer =[
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]

def flatten(xss:list[list[Any]]) -> list[Any]:
    return [x for xs in xss for x in xs]

extract_label = lambda x:Path(x).parts[-2]

class Img_Dataset(Dataset):
    
    def __init__(self, img_path:list[os.PathLike], label_map:dict[str, int], normalizer:list[Callable]=G_normalizer, dsize:Optional[tuple[int, int]] = None) -> None:
        super().__init__()
        self.dsize = dsize
        self.T = [transforms.Resize(dsize)] + normalizer if dsize is not None else normalizer
        self.T = transforms.Compose(self.T)

        self.ndata = len(img_path)
        self.img_path = img_path

        self.label_cls_map = label_map.copy()
        self.cls_label_map:dict[int, str] = {v:k for k,v in label_map.items()}
        self.ncls = len(self.cls_label_map)

        self.dtype:str="basic"
        
        self.label, self.cls_count = self._get_label(imgpath=self.img_path) 
        self.cls_w = self.ndata/self.cls_count

    def _get_label(self, imgpath:list[os.PathLike]) -> tuple[torch.Tensor, torch.Tensor]:
        
        label = torch.tensor(
            [self.label_cls_map[extract_label(i)] for i in imgpath],
            dtype=torch.long
        )
        v, cnt = torch.unique(label, return_counts=True)
        cnt = (cnt[torch.argsort(v)]).to(dtype=torch.float)
        return label, cnt
    
    def __len__(self) -> int:
        return self.ndata
    
    def _imread(self, imgpath:str)->torch.Tensor:
        return self.T(Image.open(imgpath).convert("L"))
    
    def __repr__(self) -> str:
        return "basic, should not use"

    @classmethod
    def build_datasets(cls, file_table:dict[str, dict[str, Any]], label_map:dict[str, int], logger:Optional[Logger]=None, dsize:Optional[tuple[int,int]] = None) -> dict[str, "Img_Dataset"]:
        
        table = {'train' :[], 'valid':[],'test':[]}
        for k, v in file_table.items():
            for task, vi in v.items():
                table[task] += vi
        
        print(label_map)
        d = {k:cls(img_path=v, label_map=label_map, dsize=dsize) for k, v in table.items() if len(v)}
        
        for k, v in d.items():
            if logger is None:
                print(f"-- {k} : {v} --")
            else:
                logger.info(f"-- {k} : {v} --")
        return d

class FullImg_Dataset(Img_Dataset):

    def __init__(self, img_path:list[os.PathLike], label_map:dict[str, int], **kwargs) -> None:
        
        super().__init__(img_path=img_path, label_map=label_map, normalizer=G_normalizer, dsize=kwargs.get('dsize', None))
        
        self.dtype = "img"
        
    def __getitem__(self, index) -> tuple[os.PathLike, torch.Tensor, torch.Tensor]:
        im = self._imread(self.img_path[index])
        #print(self.img_path[index], im.size())
        return (self.img_path[index], im, self.label[index])
    
    def __repr__(self) -> str:
        return f"full {self.dsize} image dataset containing {self.cls_count} for {list(self.cls_label_map.values())} classes"
    
class Inference_Dataset(Dataset):
    def __init__(self, root:Path, normalizer:list=G_normalizer, dsize:Optional[tuple[int, int]]=None) -> None:
        super().__init__()
        self.dsize = dsize
        self.T = [transforms.Resize(dsize)] + normalizer if dsize is not None else normalizer
        self.T = transforms.Compose(self.T)
        self.imgs = [_ for _ in root.glob("*.jpg")]
        self.imgs.sort()
    
    def __getitem__(self, index) ->tuple[Path, torch.Tensor, int]:
        return str(self.imgs[index]) ,self.T(Image.open(self.imgs[index]).convert("L")), -1
    
    def __len__(self) -> int:
        return len(self.imgs)
