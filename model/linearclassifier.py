import torch
import torch.nn as nn

class LC(torch.nn.Module):
    
    def __init__(self, in_dim, ncls, hid:int=1024, hidden_layers:int=2) -> None:
    
        super().__init__()
        
        self.input_layer = nn.Linear(in_features=in_dim, out_features=hid)
        self.bn1 = nn.BatchNorm1d(num_features=hid)
        self.act1 = nn.LeakyReLU()
        self.hidden = None if hidden_layers == 0 else\
            nn.Sequential(
                *[
                    self._make_a_layer(d = hid)
                    for _ in range(hidden_layers)
                ]
            )
    
        self.out_layer = nn.Linear(in_features=hid, out_features=ncls)
    
    def _make_a_layer(self, d:int)->nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=d, out_features=d),
            nn.BatchNorm1d(num_features=d),
            nn.LeakyReLU()
        ) 

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        xout = self.input_layer(x)
        xout = self.bn1(xout)
        if self.hidden is not None:
            xout = self.hidden(xout)
        xout = self.act1(xout)
        
        return self.out_layer(xout)