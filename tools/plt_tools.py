import gc
import os
from typing import Any, Optional
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_curves(lst:list[tuple[list, str]], plt_title:str, saveto:Path):
    
    f = plt.figure(dpi=600)
    plt.grid(True)
    
    for t, l in lst:
        plt.plot(np.arange(len(t))+1, t, label=l)
    
    plt.title(plt_title)
    plt.legend()
    
    plt.savefig(saveto)
    plt.close()

def draw_boxes(bg:np.ndarray, boxes:dict[str, Any], boxID:bool=False, save_to:Optional[os.PathLike]=None, save_log:bool=False) -> np.ndarray|None:
    
    draw = bg.copy()
    
    if draw.ndim == 2:
        draw = cv2.cvtColor(draw, cv2.COLOR_GRAY2BGR)
    
    for idx, bi in enumerate(boxes):
        if boxID:
            cv2.putText(
                draw,f"{idx}",
                org=(bi['xyxy'][1], max(bi['xyxy'][0] - 5, 0)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, color=(255, 255, 255), thickness=1
            )      
        cv2.rectangle(
            draw, 
            (bi['xyxy'][1], bi['xyxy'][0]), 
            (bi['xyxy'][3], bi['xyxy'][2]),
            color= (0,0,255), thickness = 1
        )
    if save_to is not None:
        
        write_ret = cv2.imwrite(str(save_to), draw)
        if save_log:
            print(f"{save_to} : {write_ret}")
        
        del draw 
        gc.collect()
        
        return None 

    return draw