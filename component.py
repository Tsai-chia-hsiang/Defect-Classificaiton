import cv2
from typing import Iterable
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tools.ccblob import ConnectedComponetBlob, detect_defect
from tqdm import tqdm

blober = ConnectedComponetBlob(
    blob_area_lowerbound = 110, 
    min_pixels_per_comp = 70, 
    region_mean = 20,
    box_gray_lowerbound = 25
)

noob_blober =  ConnectedComponetBlob(
    blob_area_lowerbound = 110, 
    min_pixels_per_comp = 0, 
    region_mean = 0,
    box_gray_lowerbound = 0
)

def check_error_only():
    error_table = []
    with open("loss.txt", "r") as f:
        for li in f.readlines():
            error_table .append(li.strip())
    
    for im in tqdm(error_table):
            
            #cv2.imwrite(
            #    str(dst/f"{im.stem}.jpg"), 
            #    detect(img = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE))
            #)
        bboxs, pca, d, g, crops, draw = detect_defect(blober=blober, img = cv2.imread(str(Path(im)), cv2.IMREAD_GRAYSCALE))
        if len(bboxs) == 0:
            print(f"Error {im}")
    
def overall():
    with open("e.log", "w+", encoding="utf-8") as log:
        for t in range(1, 4):
            target = Path("dataset")/"p2"/f"Type{t}"
            dst = Path("dataset")/"det"/f"Type{t}"
            dst.mkdir(parents=True, exist_ok=True)
            target_img = [_ for _ in target.glob("*.jpg") if 'aug' not in _.stem]
            target_img.sort(key=lambda x:int(x.stem))
            for im in tqdm(target_img):
                bboxs, pca, d, g, crops= detect_defect(
                    blober=blober, 
                    img = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE), 
                    return_draw=False
                )

                if t == 0:
                    if len(bboxs) > 0:
                        print(f"Error {im}", file=log, flush=True)
                else:
                    if len(bboxs) == 0:
                        print(f"Error {im}", file=log, flush=True)
                    
if __name__ == "__main__":
    #overall()
    #check_error_only()
    bboxs, pca, d, g, crops, draw = detect_defect(
        img=cv2.imread("dataset/p2/Type2/11.jpg", cv2.IMREAD_GRAYSCALE),
        blober=blober
    )

    cv2.imwrite("v.jpg", draw)