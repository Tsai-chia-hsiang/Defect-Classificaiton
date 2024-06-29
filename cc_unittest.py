import sys
import cv2
from tqdm import tqdm
from typing import Any
import numpy as np
from pathlib import Path
from tools.dataset import extract_label
from tools.ccblob import ConnectedComponetBlob

np.random.seed(42)
ROOT = Path("dataset")/"p2"
SAVE = Path("connectedcomponents")/"vis"


noob_blober = ConnectedComponetBlob(
    blob_area_lowerbound = 200, 
    min_pixels_per_comp = 0, region_mean = 0,
    box_gray_lowerbound = 25
)

def unit_test(imgpath:Path, show_line=False):
    img0 = cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE)
    #ann = open(SAVE/f"{extract_label(imgpath)[-1]}_{imgpath.stem}.txt", "w+")
    boxs, _ = noob_blober(img=img0, need_crop=True, topk_light = 0)

    draw = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    for idx, bi in enumerate(boxs):
    
        cv2.putText(
            draw,f"{idx}",
            org=(bi['xyxy'][1], max(bi['xyxy'][0] - 5, 0)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, color=(255, 255, 255), thickness=1
        )
            
    
        cv2.rectangle(
            draw, (bi['xyxy'][1], bi['xyxy'][0]), 
            (bi['xyxy'][3],bi['xyxy'][2]),
            color= (0,0,255), thickness = 2
        )
        if len(bi['lines'][0]) and show_line:
            cv2.putText(
                draw,f"({len(bi['lines'][0])})",
                org=(bi['xyxy'][1]+ 15, max(bi['xyxy'][0] - 5, 0)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, color=(0, 255, 0), thickness=1
            )
            for li in bi['lines'][0]:                    
                start_point = (int(li[0] + bi['xyxy'][1]), int(li[1] + bi['xyxy'][0]))
                end_point = (int(li[2] + bi['xyxy'][1]), int(li[3] + bi['xyxy'][0]))
                cv2.line(draw, start_point, end_point,color = (0,255,0), thickness = 1)

    #print(f"log is at {SAVE/f'{extract_label(imgpath)[-1]}_{imgpath.stem}.txt'}")
    saving_path = str(SAVE/f"{extract_label(imgpath)[-1]}_{imgpath.stem}.jpg")
    #print(f"save to {saving_path} {cv2.imwrite(saving_path,draw)}")
    cv2.imwrite(saving_path,draw)
    

if __name__ == "__main__":
    t = f"Type{sys.argv[1]}"
    # imall = get_origin_data_files(root=Path("dataset/p2"))[t]
    f = f"{sys.argv[2]}.jpg"
    full_path = ROOT/t/f
    unit_test(imgpath=full_path ,show_line=True)