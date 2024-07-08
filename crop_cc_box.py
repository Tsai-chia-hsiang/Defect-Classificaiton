import sys
import cv2
from time import time
from tqdm import tqdm
import numpy as np
import shutil
from pathlib import Path
from tools.dataset import extract_label, get_origin_data_files
from tools.ccblob import ConnectedComponetBlob
from tools.plt_tools import draw_boxes
from tools.io import write_json, read_json

ROOT = Path("dataset")/"source"
VISSAVE = Path("connectedcomponents")/"vis"
PATCHSAVE = Path("dataset")/"patches"

defect_blober = ConnectedComponetBlob(
    blob_area_lowerbound = 180, 
    min_pixels_per_comp = 0, region_mean = 0, 
    box_light_lowerbound = 20
)

view_blober = ConnectedComponetBlob(
    blob_area_lowerbound = 100, 
    min_pixels_per_comp = 0, region_mean = 0, 
    box_light_lowerbound = 20
)

def unit_test():
    VISSAVE.mkdir(exist_ok=True, parents=True)
    imgpath = ROOT/f"Type{sys.argv[1]}"/f"{sys.argv[2]}.jpg"
    print(imgpath)
    img0 = cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE)
    s = time()
    boxes = view_blober(img=img0, need_crop=False, topk=5)
    print(time() - s)
    for bi in boxes:
        print(bi['area'], bi['avg_light'])
    draw_boxes(bg=img0, boxes=boxes, boxID=True,save_to=VISSAVE/f"{extract_label(imgpath)[-1]}_{imgpath.stem}.jpg", save_log=True)
    

def write_patches():
    
    table = read_json("table/release/release_train_vaild_test.json")
    patch_coo_table = {}
    table = {key: table[key] for key in ["Type1", "Type2", "Type3", "Type0"]}
    for ti, img in table.items():

        ti_root = ROOT/ti
        save_dir = PATCHSAVE/ti
        if save_dir.is_dir():
            shutil.rmtree(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"crop images from {ti_root} and save to {save_dir}")
        patch_coo_table[ti] = {}

        for task, imgi in img.items():
            
            print(task)
            patch_coo_table[ti][task] = []
            
            for img_ti_i in tqdm(imgi):
                src_path = str(ti_root/f"{img_ti_i}.jpg")
                boxes, crops = defect_blober(
                    img = cv2.imread(src_path , cv2.IMREAD_GRAYSCALE), 
                    need_crop=True, 
                    topk = 5
                )
                
                img_ti_i_patch = []
                
                if len(boxes) == 0 and ti != "Type0":
                    print(ti_root/f"{img_ti_i}.jpg", flush=True)
                if len(boxes) > 0 and ti == "Type0":
                    print(ti_root/f"{img_ti_i}.jpg", flush=True)

                max_area = None 

                for idx, (bi, ci) in enumerate(zip(boxes, crops)):
                    
                    if idx > 0:
                        if bi['area']/max_area < 0.3 and bi['area'] < 600: 
                            break
                    else:
                        max_area = bi['area']

                    patch_i_path = str(save_dir/f"{img_ti_i}_{idx}.jpg")
                    cv2.imwrite(patch_i_path, ci)
                    img_ti_i_patch.append([patch_i_path ,bi['xywh'].tolist()])
                        
                patch_coo_table[ti][task].append({src_path :img_ti_i_patch})
        
    write_json(patch_coo_table,PATCHSAVE/"patches.json")

 
if __name__ == "__main__":
    if len(sys.argv) == 3:
        unit_test()
    else:
        PATCHSAVE.mkdir(exist_ok=True, parents=True)
        write_patches()