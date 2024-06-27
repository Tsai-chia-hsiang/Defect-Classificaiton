import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from .io import read_json


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

class Big_Data_IMG(Dataset):
    
    def __init__(self, img_path_list:list[Path], label_map:dict[str, int], log_smooth:bool=True) -> None:
        
        super().__init__()
        self.label_cls_map = label_map.copy()
        self.cls_label_map:dict[int, str] = {v:k for k,v in label_map.items()}
        
        self.nimg = len(img_path_list)
        self.ncls = len(self.cls_label_map)
        self.cls_count = torch.zeros((self.ncls))
        
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
        return (
            self.img_with_label[index][0], 
            G_normalizor(
                Image.open(self.img_with_label[index][0]).convert("L")
            ), 
            self.img_with_label[index][1]
        )
    
    def __len__(self)->int:
        return self.nimg


class Feature_Data(Big_Data_IMG):
    
    def __init__(self, img_path_list: list[Path], label_map: dict[str, int], log_smooth: bool = True) -> None:
        super().__init__(img_path_list, label_map, log_smooth)
        self.features = np.vstack(
            [np.load(i[0]) for i in self.img_with_label]
        )
        self.features /= (np.max(np.abs(self.features), axis=1, keepdims=True)+ 1e-8)
        self.features = torch.from_numpy(self.features)
        self.features = self.features.to(dtype=torch.float32)
        
    def __getitem__(self, index) -> tuple[Path, torch.Tensor, torch.Tensor]:
        
        return (
            self.img_with_label[index][0], 
            self.features[index], 
            self.img_with_label[index][1]
        )
      


def get_datasets(file_table:dict, label_map:dict[str, int], pre_emd:bool=False, w_log_smooth:bool=True) -> dict[str, Big_Data_IMG]:
    
    table = {
        'train' :[],
        'valid':[],
        'test':[]
    }
    for k, vi in file_table.items():
        if 'train' in vi:
            table['train'] += vi['train']
        if 'valid' in vi:
            table['valid'] += vi['valid']
        if 'test' in vi:
            table['test'] += vi['test']
    
    print(label_map)
    COSTURCTOR = Big_Data_IMG if not pre_emd else Feature_Data
    return {
        k: COSTURCTOR(img_path_list=v, label_map=label_map , log_smooth=w_log_smooth)
        for k, v in table.items() if len(v)
    }


