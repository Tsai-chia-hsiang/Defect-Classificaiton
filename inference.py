from pathlib import Path
from argparse import ArgumentParser
import torch
import pandas as pd
from time import time
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tools.torchtools import set_seed
from tools.io import read_json
from model import remove_module_prefix
from model import MODEL_MAP
from torchvision import transforms
from model.classification import train, test

class Inferenec_Set(Dataset):
    def __init__(self, root:Path) -> None:
        super().__init__()
        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.imgs = [_ for _ in root.glob("*.jpg")]
        self.imgs.sort()
    def __getitem__(self, index) ->tuple[Path, torch.Tensor]:
        return str(self.imgs[index]) ,self.T(Image.open(self.imgs[index]).convert("L"))
    
    def __len__(self) -> int:
        return len(self.imgs)
    
@torch.no_grad()
def main(test_root:Path, pretrained_ckpt:Path, pred_save:Path):
    infer =Inferenec_Set(test_root)
    loader = DataLoader(infer, batch_size=100)
    model = MODEL_MAP['resnet']( grayscale=True, ncls=4, encoder_depth = '50')
    model.load_state_dict(
        remove_module_prefix(torch.load(pretrained_ckpt,map_location='cpu'))
    )
    model.eval()
    model = model.to(device=torch.device("cuda:0"))
    name_lst = []
    pred_cls = []
    for p, img in tqdm(loader):
        #print(img.size(), img.dtype)
        ypred = model(img.to(device=torch.device("cuda:0")))
        cls_id = torch.argmax(ypred, dim=1).cpu().tolist()
        name_lst += [Path(_).name for _ in p]
        pred_cls += [f"Type{_}" for _ in cls_id]
    
    pred_df = pd.DataFrame(
        {
            'filename':name_lst,
            'Type':pred_cls
        }
    )
    pred_df.to_csv(pred_save, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_root", type=Path)
    parser.add_argument("--test_root", type=Path, default=Path("dataset")/"test")
    args = parser.parse_args()
    set_seed(891122)
    main(
        test_root = args.test_root, 
        pretrained_ckpt = args.model_root/"resnet50.pt",
        pred_save =  args.model_root/"submit.csv"
    )