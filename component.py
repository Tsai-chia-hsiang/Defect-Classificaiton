import cv2
from tqdm import tqdm
from pathlib import Path
import numpy as np
from tools import ccblob
from tools.io import write_json
blober = ccblob.ConnectedComponetBlob(
    blob_area_lowerbound = 180,
    peak_lowerbound = 160
)


root = Path("dataset")

def unit_test(img_path:str, save_path:str):
    img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
    bboxes, pca_lc, d = blober(img=img)
    showed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for bi, pcai, di in zip(bboxes, pca_lc, d):
        # print(f"P = {bi['peak']:3d}, count={bi['count']:4d}, A = {bi['area']:4d}, D={bi['density']:.3f}")
        if pcai[2] > 80:
            print(f"Peak = {pcai[0]:3d}, count={pcai[1]:4d}, A = {pcai[2]:4d}, D={di}")
            
            cv2.rectangle(
                    showed, 
                    (bi[1], bi[0]),
                    (bi[3], bi[2]), 
                    color=(0,0,255), thickness=1
                )
    
    cv2.imwrite(save_path, showed)
    return len(bboxes)

def all_a_types(target_type:str):
    view_map = {}
    t1 = root/"p2"/target_type
    dst = root/"bbox_vis"/target_type
    dst.mkdir(parents=True, exist_ok=True)
    im_all = [_ for _ in t1.glob("*.jpg") if 'aug' not in str(_)]
    im_all.sort(key=lambda x:int(x.stem))
    dbox = 0
    for i in tqdm(im_all):
        save_path = dst/f"{i.stem}.jpg"
        nbi = unit_test(img_path=str(i), save_path=str(save_path))
        view_map[i.stem] = nbi
        dbox += nbi
    print(dbox)
    write_json(view_map, f"{target_type}_dbox.json")

if __name__ == "__main__":
    #all_a_types(target_type="Type3")
    unit_test(img_path="dataset/p2/Type3/575.jpg", save_path="view.jpg")
    #unit_test(img_path="dataset/p2/Type1/2.jpg", save_path="view.jpg")