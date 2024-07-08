from pathlib import Path
import argparse
from time import time
import torch
from tools.torchtools import set_seed
set_seed(seed=891122)
from tools.dataset import build_datasets, patch_agg_collate_fn
from tools.io import read_json
from model import remove_module_prefix
from model import MODEL_MAP
from model.classification import train, test

def parsing():
    p = argparse.ArgumentParser()
    
    # dataset metadata
    p.add_argument("--data_table", type=Path, default=Path("table")/"release_train.json")
    p.add_argument("--label_map", type=Path, default=Path("table")/"label.json")
    p.add_argument("--dtype", type=str, default="fullimg")
    p.add_argument("--patch_independent", action='store_true')
    p.add_argument("--coor", type=str, default="pos")

    # hyper parameters
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batchsize", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patient", type=int, default=10)
    
    # model saving path
    p.add_argument("--ckpt_dir", type=Path, default=Path("ckpt")/"baseline")
    p.add_argument("--using_model", type=str, default="resnet50")
    
    # task
    p.add_argument("--train", action='store_true')
    p.add_argument("--test", action='store_true')
    
    # classification loss
    p.add_argument("--loss", type=str, default="ce")
    p.add_argument("--focal_gamma", type=float, default=2)
    p.add_argument("--weight_loss", action='store_true')

    
    args = p.parse_args()
    return args

if __name__ == "__main__":
    
    args = parsing()
    
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # prepare dataset 
    label_map = read_json(args.label_map)
    data_table = read_json(args.data_table)
    print(f"from {args.data_table} reading all data")
    print(f"class-label map: {label_map}")
    
    if args.train:
        print(f"settings :")
        print(f"lr: {args.lr}, epochs : {args.epochs}, batchsize : {args.batchsize}")
    
    dataset = build_datasets(
        file_table = data_table, 
        label_map = label_map, 
        coor_using=args.coor,
        dtype=args.dtype, 
        patch_independent=args.patch_independent,
        src_wh=[763, 463]
    )
    
    model:torch.nn.Module = None
    
    contain_coo = False
    if 'train' in dataset:
        contain_coo = dataset['train'].contain_coo
    elif 'test' in dataset:
        contain_coo = dataset['test'].contain_coo


    if 'resnet' in args.using_model:
        depth = args.using_model[6:]
        model = MODEL_MAP['resnet'](
            grayscale=True, ncls=len(label_map), 
            coo = contain_coo, encoder_depth = depth,
            pos_emd = args.coor == "pos", 
            patch_agg = not args.patch_independent
        )

 
    collect_fn = None 
    if args.dtype == 'patch' and not args.patch_independent :
        collect_fn = patch_agg_collate_fn
    print(collect_fn)
    model_name = f"{args.using_model}.pt"

    if args.train:
        print(f"model will be saved at {ckpt_dir/model_name}")
        s = time()
        train(
            model = model, dataset=dataset, loader_collection=collect_fn, 
            ckpt_dir=ckpt_dir, model_name=model_name,
            cls_loss = args.loss, focal_gamma=args.focal_gamma,
            epochs=args.epochs, batchsize=args.batchsize, lr=args.lr, early_stop=args.patient,
            loss_weight={
                'train':dataset['train'].cls_w if args.weight_loss else None,
                'valid':dataset['valid'].cls_w if args.weight_loss else None
            },
            return_model=False
        )
        t = time()
        print(f"training using {(t-s)/60} mins.")
        
        # test
    if args.test:
        model.load_state_dict(
            remove_module_prefix(
                torch.load(ckpt_dir/model_name,map_location='cpu')
            )
        )
        print("test ")
        acc, f1, recall, cls_recall, pred_df, error_df = test(
            model = model, dev = torch.device(f"cuda:0"), 
            test_dataset = dataset['test'], 
            loader_collection=collect_fn,
            batchsize = 60
        )
        
        print(f"accuracy : {acc:.3f}, f1 : {f1:.3f}")
        print(f"recall : {recall:.3f}")
        print(cls_recall)

        error_df.to_csv(ckpt_dir/f"test_error.csv",index=False)
        pred_df.to_csv(ckpt_dir/f"test_pred.csv",index=False)
   