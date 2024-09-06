from pathlib import Path
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pandas as pd
from libs.model import ResNet_CLS_Contrastive_Model as resnext_model
from libs.dataset import Inference_Dataset
from libs.trainer import Supervised_CLS_Contrastive_Learning_Wrapper
from libs import read_json

def main(args):

    test_set = Inference_Dataset(root=args.root, dsize=args.dsize)
    label_map = read_json(args.label_map)
    label_map = {v:k for k,v in label_map.items()}
    model = resnext_model.get_model(
        ncls=len(label_map), 
        grayscale=True, 
        depth=str(args.model), 
        pretrained=args.trained_weights
    )
    infer_warpper = Supervised_CLS_Contrastive_Learning_Wrapper(
        model=model, dp=True,
        #device = torch.device(f"cuda:{args.device}" if args.device != 'cpu' else 'cpu')
    ) 
    predict = infer_warpper.inference_one_epoch(inference_loader=DataLoader(test_set, batch_size=args.batch))
    predict.sort(key=lambda x:int(Path(x[0]).stem))
    name, pred = tuple(zip(*predict))
    j = pd.DataFrame(
        {
            "filename":list(map(lambda x:Path(x).name, name)), 
            "Type":list(map(lambda x:label_map[x], pred))
        }
    )
    print(j.shape)
    j.to_csv(args.trained_weights.parent/"predict.csv", index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset")/"release"/"test")
    parser.add_argument("--model", type=str, default="rx101_32")
    parser.add_argument("--device", default=0)
    parser.add_argument("--trained_weights", type=Path, default="./ckpt/smallsize_rx10132/resnextrx101_32_e69.pt")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dsize", nargs='+', type=int, default=(256,156))
    parser.add_argument("--label_map", type=Path, default=Path("dataset/label.json"))
    args = parser.parse_args()
    main(args=args)
