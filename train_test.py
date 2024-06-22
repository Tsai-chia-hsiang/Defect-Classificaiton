from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from time import time
from tools.torchtools import set_seed
set_seed(seed=891122)
from tools.dataset import get_datasets
from tools.io import read_json
from model import remove_module_prefix
from model.resnet import resnet_18, resnet_34, resnet_50
from model.classification import train, test
from model.linearclassifier import LC
import argparse

MODEL_MAP = {
    'resnet34':resnet_34,
    'resnet18':resnet_18,
    'resnet50':resnet_50,
    'lc':LC
}
def is_metrics_col(k:str)->bool:
    
    if 'recall' in k or \
        'f1' in k or \
        'precision' in k or\
        'accuracy' in k:
    
        return True
    
    return False

def cross_validation_table(label_map:dict[str, int])->dict[str, list]:
    cv_validation_table = {}
    cv_validation_table['fold'] = []
    for l,i in label_map.items():
        cv_validation_table[f"{i}_right"] = [] 
        cv_validation_table[f"{i}_gt"] = []
        cv_validation_table[f"{i}_recall"] = []
    cv_validation_table['marco_recall'] = []
    cv_validation_table['accuracy'] = []
    cv_validation_table['marco_f1'] = []
    return cv_validation_table

def parsing():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batchsize", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="1")
    p.add_argument("--data_table", type=Path, default=Path("dataset")/"table.json")
    p.add_argument("--label_map", type=Path, default=Path("dataset")/"label.json")
    p.add_argument("--ckpt_dir", type=Path, default=Path("ckpt")/"resnet50")
    p.add_argument("--using_model", type=str, default="resnet50")
    p.add_argument("--weight_loss", action='store_true')
    p.add_argument("--valid_only", action='store_true')
    p.add_argument("--local_test", action='store_true')
    p.add_argument("--hiddensize", type=int,default=1024)
    p.add_argument("--num_hid", type=int, default=2)
    p.add_argument("--no_valid", action='store_true')
    args = p.parse_args()
    return args

if __name__ == "__main__":
    
    args = parsing()
    
    #settings
    dev = torch.device(f"cuda:{args.device}")
    
    # prepare dataset 
    label_map = read_json(args.label_map)
    data_table = read_json(args.data_table)
    print(f"from {args.data_table} reading all data")
    print(f"class-label map: {label_map}")

    print(f"settings :")
    print(f"lr: {args.lr}, epochs : {args.epochs}, batchsize : {args.batchsize}, device : {dev}")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    cv_table = cross_validation_table(label_map=label_map)

    for fi, fold in enumerate(data_table): 
        
        dataset = get_datasets(
            file_table = fold, label_map = label_map,
            pre_emd = False if 'lc' not in args.using_model else True,
            w_log_smooth = False
        )
        model_name = f"{args.using_model}_fold{fi}.pt"
        model:torch.nn.Module = None
    
        if 'resnet' in args.using_model:
            print(args.using_model)
            model = MODEL_MAP[args.using_model](
                grayscale=True, 
                ncls=dataset['train'].ncls if 'train' in dataset else dataset['test'].ncls
            )
        elif 'lc' in args.using_model:
            in_d = dataset['train'].features.size(1)
            model = MODEL_MAP[args.using_model](
                in_dim = in_d, 
                hid = args.hiddensize,
                ncls=dataset['train'].ncls,
                hidden_layers = args.num_hid
            )
        else:
            raise KeyError("Do not support this type of model")
        
        print(f"fold : {fi}")

        # train
        
        if not args.valid_only:
            print(f"model will be save at {ckpt_dir/model_name}")
            s = time()
            train(
                model = model, 
                dataset=dataset, epochs=args.epochs, ckpt_dir=ckpt_dir, 
                model_name=model_name, batchsize=args.batchsize,
                lr=args.lr, return_model=False,
                loss_weight={
                    'train':dataset['train'].cls_w if args.weight_loss else None,
                    'valid':dataset['valid'].cls_w if args.weight_loss else None
                }
            )
            t = time()
            print(f"training using {t-s} secs.")
        
        # validation 
        if not args.no_valid:
            model.load_state_dict(
                remove_module_prefix(
                    torch.load(ckpt_dir/model_name,map_location='cpu')
                )
            )
            print("validaton ")
            acc, f1, recall, cls_recall, pred_df, error_df = test(
                model = model, dev = dev, 
                test_dataset = dataset['valid'], 
                batchsize = args.batchsize
            )
        
            print(f"accuracy : {acc:.3f}, f1 : {f1:.3f}")
            print(f"recall : {recall:.3f}")
            print(cls_recall)
            
            cv_table['fold'].append(fi)
            for type_i, r in cls_recall.items():
                cv_table[f"{type_i}_right"].append(r[0])
                cv_table[f"{type_i}_gt"].append(r[1])
                cv_table[f"{type_i}_recall"].append(r[2])
            cv_table['marco_recall'].append(recall)
            cv_table['accuracy'].append(acc)
            cv_table['marco_f1'].append(f1)
            

            if not args.local_test :
                error_df.to_csv(ckpt_dir/f"{fi}_valid_error.csv",index=False)
                pred_df.to_csv(ckpt_dir/f"{fi}_valid_pred.csv",index=False)
            else:
                pred_df.to_csv("local.csv",index=False)

    if len(cv_table) and not args.no_valid:
        
        for k in cv_table:
            if k == "fold":
                cv_table[k].append("avg")
                continue
            mean_metrics = np.mean(cv_table[k]) if is_metrics_col(k) else "-"
            cv_table[k].append(mean_metrics)
   
        cv_table = pd.DataFrame(cv_table)
        cv_table.to_csv(ckpt_dir/f"valid_metrics.csv",index=False)        