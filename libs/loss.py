from typing import Callable
import torch
import torch.nn.functional as F

def numerical_stable(logits:torch.Tensor) -> torch.Tensor:
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    return logits - logits_max.detach()

class LogitAdjust(Callable):

    def __init__(self, cls_num_list:torch.Tensor, tau=1, weight=None, device:torch.device=torch.device("cpu")):
        super().__init__()
        cls_num_list = cls_num_list.clone().detach()
        cls_num_list.to(device=device)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.m_list = self.m_list.to(device=device)
        self.weight = weight.to(device) if weight is not None else None
        
        self.name = "wce" if self.weight is not None else ""
        self.name += " with logit adjust by prob."

    def __call__(self, x, target)->torch.Tensor:
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

    def __repr__(self):
        return self.name 
    
class SCL(Callable):

    def __init__(self, ncls:int, temperature=0.1, device=torch.device("cpu")):
        
        super().__init__()
        self.on_device = device
        self.temperature = temperature
        self.ncls = ncls
        self.cls_indices = torch.arange(self.ncls, device=device).to(dtype=torch.int64)

    def __call__(self, features:torch.Tensor, targets:torch.Tensor, prototype:torch.Tensor) -> torch.Tensor:
        
        bs = features.size(0)
        cls_index = torch.concat([targets, self.cls_indices]).detach()
        cls_count = torch.histc(
            cls_index.to(dtype=torch.float32), 
            bins=self.ncls, min=0, max=self.ncls-1
        ).view(1, -1).detach()
        
        class_one_hot_axis = F.one_hot(cls_index, self.ncls).to(
            dtype=torch.float32, device=self.on_device
        ).detach()
        
        #size :  N x (N + CLS) with self mask 
        cosine_map = (features@torch.vstack([features,prototype]).T/self.temperature).fill_diagonal_(0)
        cosine_map = numerical_stable(cosine_map).fill_diagonal_(0)
        
        # since there are prototypes for each class, cls_count can't be 0
        # at any class index (at least 1 : prototype for that class)
        cls_avg = torch.exp(cosine_map)@class_one_hot_axis/cls_count
        cls_avg = torch.sum(cls_avg, dim=1, keepdim=True)

        # postive samples filtering and self mask
        L = (cosine_map - torch.log(cls_avg))*(class_one_hot_axis[:, targets].T).fill_diagonal_(0)
        L = torch.sum(L,dim=1, keepdim=True)/bs
        cl = -(1/(cls_count[0, targets] -1)).view(1, -1)@L
        return cl

    def __repr__(self):
        return "supervised class group contrastive loss"