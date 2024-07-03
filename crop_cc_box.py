import sys
import cv2
from time import time
from tqdm import tqdm
from typing import Any
import numpy as np
from pathlib import Path
from tools.dataset import extract_label, get_origin_data_files
from tools.ccblob import ConnectedComponetBlob
from tools.boxtools import draw_boxes
from tools.io import write_json, read_json

np.random.seed(42)
ROOT = Path("dataset")/"p2"
VISSAVE = Path("connectedcomponents")/"vis"
PATCHSAVE = Path("connectedcomponents")/"patches"

defect_blober = ConnectedComponetBlob(
    blob_area_lowerbound = 400, 
    min_pixels_per_comp = 0, region_mean = 0,
    box_gray_lowerbound = 25
)
noob_blober = ConnectedComponetBlob(
    blob_area_lowerbound = 400, 
    min_pixels_per_comp = 0, region_mean = 0,
    box_gray_lowerbound = 0
)
def unit_test():
    imgpath = ROOT/f"Type{sys.argv[1]}"/f"{sys.argv[2]}.jpg"
    print(imgpath)
    img0 = cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE)
    s = time()
    boxes = defect_blober(img=img0, need_crop=False, topk=30)
    # print(boxes)
    print(time() - s)
    draw_boxes(bg=img0, boxes=boxes, save_to=VISSAVE/f"{extract_label(imgpath)[-1]}_{imgpath.stem}.jpg", save_log=True)
    

def write_patches():
    table = read_json("dataset/train_valid_test/baseline.json")
    patch_coo_table = {}
    for ti, img in table.items():
        
        save_dir = PATCHSAVE/ti
        save_dir.mkdir(parents=True, exist_ok=True)
        print(save_dir)
        patch_coo_table[ti] = {}
        for task, imgi in img.items():
            print(task)
            patch_coo_table[ti][task] = []
            for img_ti_i in tqdm(imgi):
                img0 = cv2.imread(str(img_ti_i), cv2.IMREAD_GRAYSCALE)
                boxes, crops = defect_blober(img=img0, need_crop=True, topk = 5)
                imid = Path(img_ti_i).stem
                for idx, (bi, ci) in enumerate(zip(boxes, crops)):
                    patch_i_path = str(save_dir/f"{imid}_{idx}.jpg")
                    cv2.imwrite(patch_i_path, ci)
                    patch_coo_table[ti][task].append(
                        [patch_i_path ,bi['xywh'].tolist()]
                    )
                    
    write_json(patch_coo_table,PATCHSAVE/"patches_coo.json")

 
if __name__ == "__main__":
    VISSAVE.mkdir(exist_ok=True, parents=True)
    PATCHSAVE.mkdir(exist_ok=True, parents=True)
    #unit_test()
    write_patches()