import torch.nn as nn
import torch
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
from tools.io import read_json
from pathlib import Path
from sklearn.metrics import recall_score, accuracy_score, f1_score

def patch2src(patch_file:os.PathLike,src_root:Path) -> str:
    
    f = Path(patch_file)
    t = f.parent.parts[-1]
    imid = f.stem.split("_")[0]
    return str(src_root/t/f"{imid}.jpg")

if __name__ == "__main__":
    src_root = Path("./dataset/p2")
    data_table = read_json("dataset/train_valid_test/baseline.json")
    patch_pred = pd.read_csv("ckpt/patch/baseline34/test_pred.csv")
    patch_pred['src'] = patch_pred['file'].apply(
        patch2src, args=(src_root,)     
    )
    pred_label = []
    gt_label = []
    for t in data_table:
        test = data_table[t]["test"]
        for ti in test:
            if ti == "dataset/p2/Type0/1426.jpg":
                continue
            pred = patch_pred[patch_pred['src'] == ti]
            gt_label.append(int(t[-1]))

            if len(pred) == 0:
                pred_label.append(0)
            else:
                l, vote = np.unique(pred['pred'].values, return_counts=True)
                pred_label.append(l[np.argmax(vote)])
    
    pred_label = np.asarray(pred_label)
    gt_label = np.asarray(gt_label)       
    
    cls_recall = recall_score(gt_label, pred_label,  average=None)
    
    cls_recall_map = {}
    for ci in range(len(data_table)):
        is_ci_idx = np.where( gt_label == ci)[0]
        pred_ci = pred_label[is_ci_idx]
        correct = np.where(pred_ci == ci)[0]
        cls_recall_map[int(ci)] = (
            len(correct), len(is_ci_idx), 
            cls_recall[ci]
        )
        
    print(cls_recall_map)
    print(f"marco f1 : {f1_score(gt_label, pred_label, average='macro')}")