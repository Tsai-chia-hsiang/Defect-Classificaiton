import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tools.ccblob import ConnectedComponetBlob
from tqdm import tqdm

blober = ConnectedComponetBlob(
    pixel_value_thr = 0,
    blob_area_lowerbound = 110, 
    min_pixels_per_comp = 70, 
    region_mean = 20
)

def detect(img:np.ndarray, return_draw=True):
    
    bboxs, pca, d, g = blober(img=img)
 
    if not return_draw:
        if len(bboxs) == 0:
            return 0
        return np.where(g > 25)[0].shape[0]
    
    draw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for bi, pcai, di, gi in zip(bboxs, pca, d, g):
        print(f"peak : {pcai[0]:3d}, {pcai[1]:3d}/{pcai[2]:3d}= {di:.3f}, L={gi:.3f}")
        if gi > 25:
            cv2.rectangle(
                draw, (bi[1], bi[0]), (bi[3], bi[2]),
                color=(0,0,255), thickness = 1
            )
    return draw


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
        x = detect(img = cv2.imread(str(Path(im)), cv2.IMREAD_GRAYSCALE))
        if x == 0:
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
                x = detect(img = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE), return_draw=False)
                if t == 0:
                    if x > 0:
                        print(f"Error {im}", file=log, flush=True)
                else:
                    if x == 0:
                        print(f"Error {im}", file=log, flush=True)
                    
if __name__ == "__main__":
    overall()
    #check_error_only()
    #x = detect(img=cv2.imread("dataset\\p2\\Type2\\128.jpg", cv2.IMREAD_GRAYSCALE))

    #cv2.imwrite("v.jpg", x)