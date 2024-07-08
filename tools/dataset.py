import os
from typing import Any, Callable, Optional
from typing import Literal
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from .boxtools import normalize_box

def flatten(xss:list[list[Any]]) -> list[Any]:
    return [x for xs in xss for x in xs]

def get_origin_data_files(root:Path, only_origin=True) -> dict[str, list[Path]]:
    
    def is_origin(x:str):
        return 'aug' not in x

    
    folders = [ _ for _ in root.iterdir() if _.is_dir()]
    return {
        fi.parts[-1]:sorted(
            [_ for _ in fi.glob("*.jpg") \
            if only_origin and is_origin(_.stem)], 
            key = lambda x: int(x.stem) if 'aug' not in x.stem else x.stem
        )
        for fi in folders
    }

extract_label = lambda x:Path(x).parts[-2]

G_normalizor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

G_patch_normalizor = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)


class _Img_Dataset(Dataset):
    
    def __init__(self, img_path:list[os.PathLike], label_map:dict[str, int], T:Callable) -> None:
        super().__init__()
        self.T:Callable = T
        self.ndata = len(img_path)
        self.img_path = img_path
        self.contain_coo = False
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

class FullImg_Dataset(_Img_Dataset):

    def __init__(self, img_path:list[os.PathLike], label_map:dict[str, int], **kwargs) -> None:
        
        super().__init__(img_path=img_path, label_map=label_map, T=G_normalizor)
        
        self.dtype = "img"
        
    def __getitem__(self, index) -> tuple[os.PathLike, torch.Tensor, torch.Tensor]:
    
        return (
            self.img_path[index], 
            self._imread(self.img_path[index]), 
            self.label[index]
        )
    
    def __repr__(self) -> str:
        return "full image dataset"

class PatchImg_Dataset(_Img_Dataset):
    
    def __init__(
        self, patch_path: list[os.PathLike], label_map: dict[str, int], 
        img_path:Optional[list[os.PathLike]],
        coor:Optional[list[torch.Tensor]],
        independent:bool=False,  
        coor_using:Literal['channel', "pos"] = "pos"
    ) -> None:
       
        super().__init__(
            img_path = flatten(patch_path) if independent else img_path, 
            label_map=label_map, T=G_patch_normalizor
        )

        self.independent = independent

        # for aggregate patches using
        self.src_path = img_path if not independent else None
        self.patch_path = patch_path if not independent else None
        
        
        self.contain_coo =True
        self.coor = flatten(coor) if independent else coor
        self.coor_using = coor_using

    def _fetch_one_patch(self, patch_path:os.PathLike, c:torch.Tensor)->torch.Tensor|tuple[torch.Tensor, torch.Tensor]:
        
        if self.coor_using == "channel":
            return PatchImg_Dataset.extend_coor(img=self._imread(imgpath=patch_path), coo=c) 
        
        elif self.coor_using == "pos":
            return self._imread(imgpath=patch_path), c
    
    def __getitem__(self, index) -> tuple[os.PathLike, torch.Tensor, torch.Tensor]|tuple[os.PathLike, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        p = None 
        c = None
        if not self.independent:
            # aggregate all patches for an image
            patch = [
                self._fetch_one_patch(pi, ci) for pi, ci in 
                zip(self.patch_path[index], self.coor[index])
            ]
            if self.coor_using == "pos":
                p = torch.stack([_[0] for _ in patch]) 
                c = torch.stack([_[1] for _ in patch])
            else:
        
                p = torch.stack(patch)
        else:
            
            patch = self._fetch_one_patch(self.img_path[index], c=self.coor[index])
            if self.coor_using == "pos":
                p = patch[0]
                c = patch[1]
            else:
                p = patch
        
        index_img_path = self.src_path[index] if self.src_path is not None else self.img_path[index]
        
        if self.coor_using == "pos": 
        
            return index_img_path, (p, c), self.label[index]
        
        # print(index_img_path, p.size(), self.label[index])

        return index_img_path, p, self.label[index]
    
    def __repr__(self) -> str:
        return f"{'independent' if self.independent else 'agg'} patch image dataset with coo {self.coor_using}"
    
    @staticmethod
    def extend_coor(img:torch.Tensor, coo:torch.Tensor) -> torch.Tensor:
        x = coo.unsqueeze(1).unsqueeze(2)
        x = x.expand(-1, img.size(1), img.size(2))
        return torch.cat((img, x), dim=0)

def patch_agg_collate_fn(batch)->tuple[tuple[Path], Any, torch.Tensor]:
    """
    Returns
    ------
    imgpath, [xi], label
    """
    src_path = tuple([i[0] for i in batch])
    patches = list(i[1] for i in batch)
    label = torch.stack([i[2] for i in batch])
    
    return src_path, patches, label


_DTYPE_DATASET_MAP = {
    "fullimg":FullImg_Dataset,
    "patch":PatchImg_Dataset
}

def build_datasets(
    file_table:dict[str, dict[str, Any]], label_map:dict[str, int],
    dtype:Literal["fullimg", "patch"]="fullimg", 
    coor_using:Literal["pos", "channel"] = "pos",
    patch_independent:bool=False, src_wh:list[float] = None
) -> dict[str, _Img_Dataset]:
    
    table = {'train' :[], 'valid':[],'test':[]}
    coo_table = {'train':[], 'valid':[], 'test':[]}
    patch_table = {'train':[], 'valid':[], 'test':[]}
    coo_flag = False
    for k, v in file_table.items():
        for task, vi in v.items():
            if isinstance(vi[0], str):
                table[task] += vi
            elif isinstance(vi[0], dict):
                coo_flag = True
                for img_patch in vi:
                    src_img_path = list(img_patch.keys())[0]
                    if len(img_patch[src_img_path]):
                        table[task].append(src_img_path)
                        patch_table[task].append([_[0] for _  in img_patch[src_img_path]])
                        coo_table[task].append([_[1] for _  in img_patch[src_img_path]])
    for task in coo_table:
        if coo_flag:
            coo_table[task]=[normalize_box(torch.tensor(i), *src_wh) for i in coo_table[task] ]

    print(label_map)

    d = {
        k: _DTYPE_DATASET_MAP[dtype](
            img_path=v, 
            label_map=label_map , 
            coor=coo_table[k],
            patch_path=patch_table[k],
            independent = patch_independent,
            coor_using = coor_using
        )
        for k, v in table.items() if len(v)
    }
    for k, v in d.items():
        print(f"-- {k} : {v} --")

    return d
