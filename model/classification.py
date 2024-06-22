from typing import Callable, Any
from tqdm import tqdm
from time import time
from copy import deepcopy
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss 
from torch.optim import Optimizer, Adam
from tools.dataset import Big_Data_IMG, G_normalizor, extract_label
from tools.plt_tools import plot_curves

def forward_one_epoch(
    model:nn.Module, loader:DataLoader, device:torch.device, 
    criteria:Callable, optr:Optimizer=None, 
    each_cls_recall:bool=False, return_prediction:bool=False
)-> tuple:
    """
    
    Return
    ------
    (loss, accuracy, f1, recall)

    if each_cls_recall = True, will attach a dict for each class recall
    
    if return_prediction = True, will attach a dict for gt as well prediction and file path

    """

    total_loss = 0
    pred_gt_df = {'gt':[], 'pred':[], 'file':[]}
    
    if optr is not None:
        # train
        model.train()
        torch.set_grad_enabled(True)
    else:
        # test
        model.eval()
        torch.set_grad_enabled(False)

    for pi, ti, li in tqdm(loader):
        
        if optr is not None:
            optr.zero_grad()
        
        y = model(ti.to(device=device))

        if criteria is not None:
            loss:torch.Tensor = criteria(y, li.to(device=device))
            total_loss += loss.item()
            if optr is not None:
                # on training
                loss.backward()
                optr.step()
        
        predicted = torch.argmax(y, dim=1).cpu()
        pred_gt_df['pred'] += predicted.tolist()
        pred_gt_df['gt'] += li.tolist()
        pred_gt_df['file'] += list(pi)


    ret = [
        total_loss/len(loader), 
        accuracy_score(y_true=pred_gt_df['gt'], y_pred=pred_gt_df['pred']), 
        f1_score(y_true=pred_gt_df['gt'], y_pred=pred_gt_df['pred'], average='macro'), 
        recall_score(y_true=pred_gt_df['gt'], y_pred=pred_gt_df['pred'], average='macro')
    ]
    if each_cls_recall:

        gt_array = np.array(pred_gt_df['gt'], dtype=np.int32)
        pred_array = np.array(pred_gt_df['pred'], dtype=np.int32)
        
        cls_recall = recall_score(y_true=pred_gt_df['gt'], y_pred=pred_gt_df['pred'], average=None)
        cls_recall_map = {}
        
        for ci in range(loader.dataset.ncls):
            is_ci_idx = np.where(gt_array == ci)[0]
            pred_ci = pred_array[is_ci_idx]
            correct = np.where(pred_ci == ci)[0]
            cls_recall_map[int(ci)] = (
                len(correct), len(is_ci_idx), 
                cls_recall[ci]
            )
        
        ret.append(cls_recall_map)
    
    if return_prediction:
        ret.append(pred_gt_df)    
    
    
    return tuple(ret)


def train(model:torch.nn.Module, dataset:dict[str, Big_Data_IMG], epochs:int = 20, batchsize:int = 40, lr:float=1e-3, loss_weight:dict[str, torch.Tensor]=None, ckpt_dir:Path=Path("ckpt"), model_name:str="model", return_model:bool=True) -> nn.Module | None:
    
    def print_cls_metrics(t:dict):
        for k,v in t.items():
            print(f"{k} : {v[0]}/{v[1]}={v[2]:.3f}")

    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(0)
    
    loader = {
        mode:DataLoader(dataset=dataset[mode], batch_size=batchsize, shuffle=True, generator=g)
        for mode in ['train', 'valid']
    }

    critieras = {
        mode: CrossEntropyLoss() if loss_weight[mode] is None 
        else CrossEntropyLoss(weight=loss_weight[mode].to(device=dev))
        for mode in ['train', 'valid']
    }

    print(f"train data : {dataset['train'].cls_count}")
    print(f"valid data : {dataset['valid'].cls_count}")
    print(f"lossw = {critieras['train'].weight}, {critieras['valid'].weight}")

    M_GPU = False
    model = model.to(device=dev)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        M_GPU = True
        model = nn.DataParallel(model)
    
    optr = Adam(params=model.parameters(), lr=lr)
        
    metrics = {
        mode:{
            'loss':[100]*epochs,
            'accuracy':[0.0]*epochs,
            'macro_f1':[0.0]*epochs,
            'macro_recall':[0.0]*epochs
        } for mode in ['train', 'valid']
    }
    
    best_val_metrics = 0
    for e in range(epochs):
        
        for mode in ['train', 'valid']:
            e_start = time()

            metrics[mode]['loss'][e], metrics[mode]['accuracy'][e],\
            metrics[mode]['macro_f1'][e], metrics[mode]['macro_recall'][e],\
            cls_recall = forward_one_epoch(
                model=model, loader=loader[mode], 
                criteria=critieras[mode], 
                device=dev, 
                optr=optr if mode == "train" else None,
                each_cls_recall=True
            )
            e_end = time()
            saved = "train"
            if mode == "valid":
                saved = False
                if metrics[mode]['macro_f1'][e] >= best_val_metrics:
                    saved = True
                    torch.save(
                        model.module.state_dict() if M_GPU else model.state_dict(), 
                        ckpt_dir/model_name
                    )
                    best_val_metrics = metrics[mode]['macro_f1'][e]

            print(f"{mode} {e} time: {e_end - e_start:.3f} secs | loss : {metrics[mode]['loss'][e]:.3f} | acc: {metrics[mode]['accuracy'][e]:.3f}")
            print(f"f1 : {metrics[mode]['macro_f1'][e]:.3f}, best f1 : {best_val_metrics:.3f}, save model : {saved}")
            print(f"recall: {metrics[mode]['macro_recall'][e]:.3f}")
            print_cls_metrics(cls_recall)
        
        print(f"=="*50)
    
    pure_name = Path(model_name).stem
    plot_curves(
        [(metrics['train']['loss'], "train")], 
        plt_title="Loss", saveto=ckpt_dir/f"{pure_name}_train_loss.jpg"
    )
    plot_curves(
        [
            (metrics['valid']['accuracy'], "val_acc"),
            (metrics['valid']['macro_f1'], "val_f1"),
            (metrics['valid']['macro_recall'], "val_recall")
        ], 
        plt_title = "Metrics", 
        saveto = ckpt_dir/f"{pure_name}_val_metrics.jpg"
    )
    
    if return_model:
        model = model.to(torch.device('cpu'))
        model.load_state_dict(torch.load(ckpt_dir/model_name, map_location='cpu'))

    return model


@torch.no_grad()
def test(model:nn.Module, test_dataset:Big_Data_IMG, dev, batchsize = 40) -> tuple[float, float, float, dict[int, tuple], pd.DataFrame, pd.DataFrame]:
    
    model = model.eval().to(device=dev)
    testloader = DataLoader(dataset=test_dataset, batch_size=batchsize) 
    criteria = CrossEntropyLoss()
    _, acc, f1, recall, cls_recall, pred_df = forward_one_epoch(
        model = model, loader = testloader, device=dev,
        criteria = criteria, return_prediction=True, 
        each_cls_recall=True
    )
    pred_df = pd.DataFrame(pred_df)
    error_df = pred_df.copy()
    error_df =  error_df[error_df['pred'] != error_df['gt']]
    pred_df.sort_values(by='gt', ascending=True, inplace=True)
    error_df.sort_values(by='gt', ascending=True, inplace=True)
    return acc, f1, recall, cls_recall, pred_df, error_df

@torch.no_grad()
def unit_test(model:nn.Module, test_img:Path, dev:torch.device, label_map:dict[int, str])->tuple[str, int, str]:
    model = model.eval().to(device=dev)
    label_map_reverse = {v:k for k,v in label_map.items()}
    gt = extract_label(test_img)
    x = G_normalizor(Image.open(test_img).convert("L")).unsqueeze(0).to(device=dev)
    pred = model(x)
    
    l = torch.argmax(pred, 1).cpu().item()
    print(pred, l)
    return gt, l, label_map_reverse[l]

