import os
from typing import Literal
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from .io import read_json
from .boxtools import normalize_box


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
class Big_Data_IMG(Dataset):
    
    def __init__(self, img_path_list:list[Path], label_map:dict[str, int], log_smooth:bool=True, **kwargs) -> None:
        
        super().__init__()
        self.label_cls_map = label_map.copy()
        self.cls_label_map:dict[int, str] = {v:k for k,v in label_map.items()}
        self.contain_coo = False
        self.nimg = len(img_path_list)
        self.ncls = len(self.cls_label_map)
        self.cls_count = torch.zeros((self.ncls))
        self.datatype = "img"
        self.img_with_label = [
            (i,  self.__get_and_accumu_cls(label=i)) 
            for i in img_path_list
        ]
        self.cls_w = self.nimg/self.cls_count
        if log_smooth:
            self.cls_w = torch.log(self.cls_w)
        
    def __get_and_accumu_cls(self, label:str)->torch.Tensor:
        
        ci =  self.label_cls_map[extract_label(label)]
        self.cls_count[ci] += 1
        
        return torch.tensor(ci, dtype=torch.long)
    
    def __getitem__(self, index) -> tuple[Path, torch.Tensor, torch.Tensor]:

        # assert self.cls_label_map[self.img_with_label[index][1].item()] == extract_label(self.img_with_label[index][0]) 
        img = G_normalizor(
            Image.open(self.img_with_label[index][0]).convert("L")
        )
        return (self.img_with_label[index][0], img, self.img_with_label[index][1])
    
    def __len__(self)->int:
        return self.nimg
    
    def __repr__(self) -> str:
        return "full image dataset"

class Patch_IMG(Big_Data_IMG):
    
    def __init__(self, img_path_list: list[Path], label_map: dict[str, int], coordinate:torch.Tensor = None, log_smooth: bool = True) -> None:
        super().__init__(img_path_list, label_map, log_smooth)
        self.coo = coordinate if coordinate is not None else None
        self.contain_coo = coordinate is not None
    
    def __getitem__(self, index) -> tuple[Path, torch.Tensor, torch.Tensor, torch.Tensor]|tuple[Path, torch.Tensor, torch.Tensor]:
        
        img = G_patch_normalizor(
            Image.open(self.img_with_label[index][0]).convert("L")
        )
        
        if  self.contain_coo:
            return (self.img_with_label[index][0], img, self.coo[index],self.img_with_label[index][1])
        
        return (self.img_with_label[index][0], img, self.img_with_label[index][1])
    def __repr__(self) -> str:
        return "patch image dataset" if not self.contain_coo else "patch image with coo dataset" 


class Feature_Data(Big_Data_IMG):
    
    def __init__(self, img_path_list: list[Path], label_map: dict[str, int], log_smooth: bool = True) -> None:
        super().__init__(img_path_list, label_map, log_smooth)
        self.features = np.vstack([np.load(i[0]) for i in self.img_with_label])
        self.features /= (np.max(np.abs(self.features), axis=1, keepdims=True)+ 1e-8)
        self.features = torch.from_numpy(self.features)
        self.features = self.features.to(dtype=torch.float32)
        
    def __getitem__(self, index) -> tuple[Path, torch.Tensor, torch.Tensor]:
        
        return (
            self.img_with_label[index][0], 
            self.features[index], 
            self.img_with_label[index][1]
        )
      


_DTYPE_DATASET_MAP = {
    "fullimg":Big_Data_IMG,
    "patchimg":Patch_IMG,
    "feature":Feature_Data
}

def build_datasets(
    file_table:dict, label_map:dict[str, int], 
    dtype:Literal["fullimg", "patch", "feature"]="fullimg", 
    w_log_smooth:bool=True, src_wh:list[float] = None
) -> dict[str, Big_Data_IMG]:
    
    table = {'train' :[], 'valid':[],'test':[]}
    coo_table = {'train':[], 'valid':[], 'test':[]}
    coo_flag = False
    for k, v in file_table.items():
        for task, vi in v.items():
            if isinstance(vi[0], str):
                table[task] += vi
            elif isinstance(vi[0], list):
                coo_flag = True
                table[ task] += [_[0] for _ in vi]
                coo_table[task] += [_[1] for _ in vi]
    for task in coo_table:
        coo_table[task] = normalize_box(
            torch.tensor(coo_table[task]), *src_wh
        ) if coo_flag else None
    
    print(label_map)

    d = {
        k: _DTYPE_DATASET_MAP[dtype](
            img_path_list=v, 
            label_map=label_map , 
            log_smooth=w_log_smooth, 
            coordinate=coo_table[k]
        )
        for k, v in table.items() if len(v)
    }
    for k, v in d.items():
        print(f"-- {k} : {v} --")

    return d


