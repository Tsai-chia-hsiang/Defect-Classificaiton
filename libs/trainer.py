import os
import os.path as osp
from typing import Optional, Literal, Any, Callable
import logging
from pathlib import Path
from tqdm import tqdm, trange
from logging import Logger
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer, Adam, AdamW, SGD
from .loss import LogitAdjust, SCL
from .evaluation import classification as eva_cls

class Supervised_CLS_Contrastive_Learning_Wrapper():

    def __init__(
        self, model:torch.nn.Module, logger:Logger=None, board:Optional[SummaryWriter]=None,
        device:torch.device=torch.device("cuda:0"), dp:bool=False,
    ) -> None:
      
        self.logger = logger
        self.board = board
        
        if dp:
            if self.logger is not None:
                self.logger.info(msg="set dp=True will ignore the device cpu setting and directly using torch.nn.DataParallel with torch.device('cuda')")
            else:
                print("set dp=True will ignore the device cpu setting and directly using torch.nn.DataParallel with torch.device('cuda')")
        self.model = model
        self.device = device if not dp else torch.device("cuda")
        self.ncls = self.model.ncls
        self.model.to(device=self.device)
        
        self.is_dp = dp
        if self.is_dp:
            self.model = torch.nn.DataParallel(self.model)
            
        
        self.optr:Optimizer = None
        self.epochs:int = None
        self.warm_up:int = None
        self.loss: dict[str, dict[str, Any]] = {
            'cls': {
                'func': None,  # This should later be set to a Callable
                'w': 0.0  # This should be a float
            },
            'cl': {
                'func': None,  # This should later be set to a Callable
                'w': 0.0  # This should be a float
            }
        }
       
        self.ckpt_dir:Path = None
        self.model_name:str= None
        self.train_loader:DataLoader=None
        self.valid_loader:Optional[DataLoader] = None

    def _set_optimizer(self, opt:Literal["adam","adamw", "sgd"]="adamw", lr:float=0.01, momentum:float=0.9)->Optimizer:

        match opt.lower():
            case "adam":
                return Adam(self.model.parameters(), lr=lr)
            case "adamw":
                return AdamW(self.model.parameters(), lr=lr)
            case "sgd":
                return SGD(self.model.parameters(), lr=lr, momentum=momentum)
            case _:
                raise KeyError(f"Not support {opt}")

    def _set_loss(self, loss_type:Literal['cls', 'cl'], func:Callable[..., torch.Tensor], w:float)->None:
        self.loss[loss_type]['func'] = func
        self.loss[loss_type]['w'] = w

    def before_train_setting(self, ckpt_dir:Path, model_name:str, train_set:Dataset, val_set:Dataset,batch:int=50 , g_seed:int=0):  
        
        """
        for ckpt dir and dataloader
        """
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        g = torch.Generator()
        g.manual_seed(g_seed)
        self.train_loader = DataLoader(dataset=train_set, batch_size=batch['train'], shuffle=True, pin_memory=True, generator=g)
        self.valid_loader = DataLoader(dataset=val_set, batch_size=batch['valid'], shuffle=False, pin_memory=True) if val_set is not None else None

    def train_model(
        self, lr:float=0.01, 
        epochs:int=20, warm_up:int=0, 
        optr:Literal["adam","adamw", "sgd"]="adamw",
        val_epochs:int=1, debug=-1, 
        cls_weight=2, cl_weight=0.6
    ) -> Path:
        
        self._set_loss(
            loss_type='cls', 
            func=LogitAdjust(
                cls_num_list=self.train_loader.dataset.cls_count, 
                device=self.device, 
                weight=self.train_loader.dataset.cls_w
            ), 
            w=cls_weight
        )
        
        self._set_loss(loss_type='cl', func=SCL(ncls=self.ncls, device=self.device), w=cl_weight)
        self.epochs = epochs
        self.warm_up = warm_up
        self.optr = self._set_optimizer(opt=optr, lr=lr)
    
        self.logger.info(f"Training classification {self.epochs} epochs with WCE and contrastive loss")

        if self.valid_loader is None:
            self.logger.info("===> No validation set, using training loss to judeg <===")
        
        best_f1 = 0
        best_loss = np.inf
        save_to:Path = None
        for e in range(epochs):
        
            if debug > 0:
                self.logger.info(f"debugging, run just {debug} batch(s)")
            
            if e < warm_up:
                self.logger.info(f"training epoch {e}, warm-up : only classification loss")
            else :
                self.logger.info(f"training epoch {e}, with classification loss and contrastive loss")

            loss_log = self._train_one_epoch(using_cl=e >= warm_up, debug_iter=debug, epoch=e)
            self.logger.info(f"training loss : {loss_log}")
    
            if self.valid_loader is not None and ( (e+1)%val_epochs == 0 or (e+1) == warm_up):
                
                self.logger.info(f"validation epoch {e}")
                valid_log = self.inference_one_epoch(inference_loader=self.valid_loader, return_pred=False, debug_iter=debug)

                self.logger.info(f"validation F1 : {valid_log['f1']}; macro : {valid_log['macro f1']}")
                
                if self.board is not None:
                    self.board.add_scalar(f"valid_macro_f1", valid_log['macro f1'], e)
                
                if valid_log['macro f1'] >= best_f1:
                    save_to = self.ckpt_dir/f'{self.model_name}_e{e}.pt'
                    self.logger.info(f"current best valid f1:{best_f1}; new best valid f1:{valid_log['macro f1']}")
                    self.logger.info(f"save weights to {save_to}")
                    
                    if self.is_dp:
                        torch.save(self.model.module.state_dict(), save_to)
                    else:
                        torch.save(self.model.state_dict(), save_to)

                    best_f1 = valid_log['macro f1']
                
            else:
                if (e+1)%val_epochs == 0 and loss_log['total'] <= best_loss:
                    self.logger.info(f"current best loss:{best_loss}; new best loss:{loss_log['total']}")
                    self.logger.info(f"save weights to {save_to}")
                    save_to = self.ckpt_dir/f"{self.model_name}_no_valid_e{e}.pt"
                    if self.is_dp:
                        torch.save(self.model.module.state_dict(), save_to)
                    else:
                        torch.save(self.model.state_dict(), save_to)
                    
                    best_loss = loss_log['total']
        
        return save_to
    
    def _train_one_epoch(self, epoch:int, pbar:bool=True, using_cl:bool=False, debug_iter=-1) -> float|dict[str, float]:
        
        img:torch.FloatTensor = None
        yi:torch.FloatTensor = None
        fi:torch.FloatTensor = None
        li:torch.LongTensor = None
        n_sample = 0
        self.model.train()
        critera_cls, critera_cl, critera_total = 0, 0, 0

        bar = tqdm(self.train_loader) if pbar else self.train_loader
        for idx, (img_path, img, li) in enumerate(bar):
            
            self.optr.zero_grad()
            li = li.to(device=self.device)
            yi, fi, pi = self.model(img.to(device=self.device), features=True)
            
            total_loss:torch.Tensor = 0.0
            cls_l:torch.Tensor = self.loss['cls']['func'](yi, li)
            feature_loss:torch.Tensor = 0.0
            
            if using_cl:
                cls_l = cls_l*self.loss['cls']['w']
                if self.is_dp:
                    pi = pi[:self.ncls]
                feature_loss = self.loss['cl']['func'](
                    features=fi, targets=li, prototype=pi
                )*self.loss['cl']['w']
            
                total_loss = cls_l + feature_loss
            else:
                total_loss = cls_l

            total_loss.backward()
            self.optr.step()
            
            n_sample += img.size(0)
            critera_cls += cls_l.item()*img.size(0)
            if using_cl:
                critera_cl += feature_loss.item()*img.size(0) 
                critera_total += total_loss.item()*img.size(0)
               
            if pbar:
                bar.set_postfix(
                    ordered_dict={
                        'cls_loss': f"{cls_l.item():.4f}",
                        'cl':f"{feature_loss.item():.4f}",
                        'total_loss':f"{total_loss.item():.4f}"
                    } if using_cl else {
                        'cls_loss' : f"{cls_l.item():.4f}"
                    }
                )
            
            if debug_iter > 0:
                if idx == debug_iter:
                    break

        critera_cls /= n_sample
        critera_cl /= n_sample
        critera_total /= n_sample
        if self.board is not None:
            self.board.add_scalar("cls_loss",critera_cls, epoch)
            if using_cl: 
                self.board.add_scalar("constrastive_loss", critera_cl, epoch)
                self.board.add_scalar("total_loss", critera_total, epoch)

        return {
            'cls':critera_cls,
            'total':critera_total
        } if not using_cl else \
        {
            'cls':critera_cls,
            'contrastive':critera_cl,
            'total':critera_total
        }
    
    @torch.no_grad()
    def inference_one_epoch(
        self, inference_loader:DataLoader,
        pbar:bool=True, return_pred:bool=True, attach_gt:bool=False,
        confusion_matrix:Optional[Literal["pd", "np"]]=None,
        metrics_tolist:bool=False,
        debug_iter:int=-1
    ) -> list|dict|tuple:
        
        self.model.eval()

        img:torch.FloatTensor = None
        li:Optional[torch.LongTensor] = None
        y:torch.FloatTensor = None
        gth:list[int] = []
        pred:list[int] = []
        imgpaths:list[str] = []
        
        bar = tqdm(inference_loader) if pbar else inference_loader
        inference_flage=False
        
        for idx, (img_path, img, li) in enumerate(bar):

            if li[0] >= 0:
                gth += li.tolist()
            else:
                inference_flage = True
                imgpaths += list(img_path)
            y = self.model(img.to(self.device))
            pred += torch.argmax(y, dim=1).cpu().tolist()
         
            if debug_iter > 0 and idx == debug_iter:
                break

        if inference_flage: # no gth, just inference
            return list(zip(imgpaths, pred))
    
        metrics = eva_cls(pred=np.array(pred), gth=np.array(gth), cm=confusion_matrix, to_pydefault_type=metrics_tolist)
        if return_pred:
            if attach_gt:
                return list(zip(imgpaths, pred, gth)), metrics
            return list(zip(imgpaths, pred)), metrics
        
        return metrics

    @torch.no_grad()
    def unit_inference(self, x:torch.Tensor)->int:
        self.model.eval()
        y = torch.argmax(self(x.unsqueeze(0).to(self.device)), dim=1).cpu()
        return y.item()
    
