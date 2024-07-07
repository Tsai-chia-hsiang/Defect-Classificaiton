import sys
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
sys.path.append("..")
import imgaug.augmenters as iaa
from tools.boxtools import is_bbox_at_edge_or_corner
from tools.ccblob import ConnectedComponetBlob
from tools.io import write_json, read_json


ROOT = Path("dataset/source")/"Type1"

def horizontally_concatenate_images_with_red_bar(images):
    # Check if the list is empty or None
    if not images or len(images) != 6:
        raise ValueError("The image list should contain exactly 6 images.")
    
    # Read the images and store them in a list
    img_list = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in images]
    
    # Get the height of the first image to maintain uniform height
    height = img_list[0].shape[0]
    
    # Create a red bar of the specified width
    red_bar = np.zeros((height, 5, 3), dtype=np.uint8)
    red_bar[:] = (0, 0, 255)  # BGR for red color in OpenCV
    
    # Separate the images into top 3 and bottom 3
    top_images = img_list[:3]
    bottom_images = img_list[3:]
    
    # Concatenate top images with the red bar in between
    top_concatenated = top_images[0]
    for img in top_images[1:]:
        top_concatenated = np.hstack((top_concatenated, red_bar, img))

    # Concatenate bottom images with the red bar in between
    bottom_concatenated = bottom_images[0]
    for img in bottom_images[1:]:
        bottom_concatenated = np.hstack((bottom_concatenated, red_bar, img))

    # Concatenate top and bottom images vertically with a red bar in between
    hred_bar = np.zeros((5, bottom_concatenated.shape[1], 3), dtype=np.uint8)
    hred_bar[:] = (0, 0, 255)  # BGR for red color in OpenCV
    final_concatenated = np.vstack((top_concatenated, hred_bar, bottom_concatenated))
    
    return final_concatenated


def pastable_top_left(xyxy:list[int], offset:int=20) -> list[tuple[int, int]]:
    
    # clockwise from 0,0

    x_offset = xyxy[2] - xyxy[0]
    y_offset = xyxy[3] - xyxy[1]
    
    # up edge →
    ret = [(0, yi) for yi in range(0, 463-y_offset, offset)]
    
    # left edge ↓
    ret += [(xi ,463-y_offset) for xi in range(0, 763-x_offset, offset)]
    
    # bottom edge ←
    ret += [(763-x_offset, yi) for yi in range(463-y_offset, 0, -offset)]

    # right edge ↑
    ret += [(xi, 0) for xi in range(763-x_offset, 0, -offset)]
    ret = list(set(ret))
    ret = [b for b in ret if abs(b[0]-xyxy[0]) > offset or abs(b[1] - xyxy[1])>offset]
    return ret

r90 = iaa.Rot90(k=1)
r270 = iaa.Rot90(k=3)
hf = iaa.Fliplr(1.0)
vf = iaa.Flipud(1.0)
FILP = {
    "l2r":hf, "r2l":hf,
    "t2b":vf, "b2t":vf,
    "l2t":r90, "t2r":r90,
    "t2l":r270, "r2t":r270,
    "l2b":r270, "b2r":r270,
    "b2l":r90, "r2b":r90
}

def filp_patch(p:np.ndarray, src_edge:list[str], dst_edge:list[str]) -> np.ndarray:
    
    filp_p = p.copy()
    for i in range(len(dst_edge)):
        trans = ""
        if dst_edge[i] is not None:
            # dst touchs this direction
            if src_edge[i] is not None and dst_edge[i] != src_edge[i]:
                # src also touch this direction
                trans = f"{src_edge[i]}2{dst_edge[i]}"
            else:
                j = 1 - i
                if dst_edge[j] is None:
                    # dst in j direction also not exist
                    if src_edge[j] is not None and dst_edge[i] != src_edge[j]:
                        trans = f"{src_edge[j]}2{dst_edge[i]}"

        if len(trans):
            filp_p = FILP[trans](image = filp_p)

    return filp_p

def crop_t1(patch_save:Path):
    
    defect_blober = ConnectedComponetBlob(
        blob_area_lowerbound = 100, 
        min_pixels_per_comp = 0, region_mean = 0,
        box_gray_lowerbound = 25
    )
    
    imall = [_ for _ in ROOT.glob("*.jpg")]
    print(len(imall))
    #imall = [Path("dataset/source/Type1/17.jpg")]
    for i in tqdm(imall):
        sample_img = cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
        major_defect, patch = defect_blober(img=sample_img, need_crop=True, topk=-1)
        
        for bi, pi in zip(major_defect, patch):
            
            if len(bi['corner']):
                past_coors = pastable_top_left(xyxy=bi['xyxy'], offset=200)
                to_past = np.random.permutation(np.arange(len(past_coors)))[:6]
                
                # mask out origin part
                sample_img[
                    bi['xyxy'][0]:bi['xyxy'][2], 
                    bi['xyxy'][1]:bi['xyxy'][3]
                ] = 0
                #print(bi['corner'])
                for order, coo_idx in enumerate(to_past):

                    coo = past_coors[coo_idx]
                    coo = np.asarray([
                        coo[0], coo[1], 
                        coo[0] + pi.shape[0], 
                        coo[1] + pi.shape[1]
                        ])
                    coo_edge = is_bbox_at_edge_or_corner(
                        coo, image_shape=sample_img.shape,
                        thr=0
                    )
                    #print(f"{coo} : {coo_edge}")
                    pasted_img = sample_img.copy()
                    pasted_img[coo[0]:coo[2], coo[1]:coo[3]] = filp_patch(
                        pi, src_edge=bi['corner'], 
                        dst_edge=coo_edge
                    )
                
                    cv2.imwrite(
                        str(patch_save/f"aug_rcp{order}_{i.stem}.jpg"), 
                        pasted_img
                    )
                    
                
                break
"""    
vis = horizontally_concatenate_images_with_red_bar(images=to_show)
cv2.imwrite(f"exp/{sample}.jpg",vis)
"""


if __name__ == "__main__":
    np.random.seed(891122)
    
    patch_save = Path("dataset/augmentation/Type1")
    patch_save.mkdir(parents=True, exist_ok=True)
    crop_t1(patch_save=patch_save)