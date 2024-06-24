import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from tools import ccblob
from tools.io import write_json

def get_a_type_imgs(root:Path, t:str)->list[Path]:
    r= [_ for _ in (root/t).glob("*.jpg") if 'aug' not in _.stem]
    r.sort(key=lambda x:int(x.stem))
    return r

blober = ccblob.ConnectedComponetBlob(
    blob_area_lowerbound = 180,
    peak_lowerbound = 180,
    density_lowerbound = 0.5
)

def unit_test(img_path:str) -> tuple[int, int]:
    img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
    bboxes, pca_lc, d = blober(img=img)
    # print(bboxes)
    if len(bboxes) == 0:
        return 0 ,0
    
    max_area = pca[:, 2].max()
    if max_area < 500:
        return 0, 0
    return len(bboxes), int(max_area)


def main():
    dataroot = Path("dataset")/"p2"
    target_type = [f"Type0", "Type1", "Type2", "Type3"]
    
    img_all:list[Path] = []
    for ti in target_type:
        img_all +=  get_a_type_imgs(dataroot, ti)

    log = open("ccp_wa.csv", "w+")
    print("files,nbox,gt,mex_area", file=log)
    pbar = tqdm(img_all)
    error_pred = 0
    for img_path in pbar:
        defect = int(img_path.parts[-2][-1]) > 0
        nbox, max_a = unit_test(img_path=str(img_path))
        is_defect = nbox > 0
        if is_defect != defect:
            error_pred += 1
            print(f"{str(img_path)},{nbox},{int(defect)},{max_a}", file=log, flush=True)
        pbar.set_postfix(ordered_dict={'error':error_pred})
    log.close()

if __name__ == "__main__":
    main()
    #nbox, a = unit_test(img_path="dataset/p2/Type2/218.jpg")
    #print(a)