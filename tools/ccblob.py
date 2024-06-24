from typing import Literal, Any, Iterable
import cv2
import numpy as np
from .boxtools import *

class ConnectedComponetBlob():

    def __init__(
        self, pixel_value_thr:int=0, 
        min_pixels_per_comp:int=20, 
        blob_area_lowerbound:int=80, 
        region_mean:float=30,
        box_gray_lowerbound:float=25
    ):
        
        self.pixel_thr = pixel_value_thr
        self.componet_lb = min_pixels_per_comp
        self.rmean = region_mean
        self.area_lb = blob_area_lowerbound
        self.box_gray_lb = box_gray_lowerbound
        self.box_thr = np.array([self.area_lb, box_gray_lowerbound])
    
    def __call__(self, img:np.ndarray, nms:bool=True, need_crop:bool=True) -> Iterable[np.ndarray]:
        
        """
        
        Returns:
        -------
        if return_type is bbox:
            return bboxes, [peaks, counts, areas, densities]
        """
        
        binary_image = (
            ( (img > self.pixel_thr).astype(np.int32) )*255
        ).astype(np.uint8)
        
        num_labels, labels_im = cv2.connectedComponents(binary_image)
  
        boxes = self.get_bbox_from_mask(img=img, num_labels=num_labels, mask=labels_im, nms=nms)
    
        # return [{'box':i} for i in boxes]
        return self._bbox_post_processing(img=img, bboxes=boxes, need_crop=need_crop)
    
    def get_bbox_from_mask(self, img:np.ndarray, num_labels:int, mask:np.ndarray, nms:bool=True) -> np.ndarray:
        
        bounding_boxes = []
        # Start from 1 to ignore the background
        for label in range(1, num_labels):  
            component_mask = (mask == label).astype(np.uint8)
            coords = np.column_stack(np.where(component_mask > 0))
            if coords.size/2 >= self.componet_lb:
                m = np.mean(img[np.where(component_mask > 0)].astype(np.float32))
                if m > self.rmean:
                    # print(m)
                    x, y, w, h = cv2.boundingRect(coords)
                    bounding_boxes.append([x, y, x + w, y + h])  

        bounding_boxes = np.asarray(bounding_boxes)
        
        return bounding_boxes if not nms \
            else non_max_suppression_fast(bounding_boxes)

    def _bbox_post_processing(self, img:np.ndarray, bboxes:np.ndarray, need_crop:bool)->Iterable[np.ndarray]:
        if len(bboxes) == 0:
            return [], [], [], []
        
        blob_crops = [crop(img=img, xyxy=bi) for bi in bboxes]
     
        areas = (bboxes[:, 2]- bboxes[: ,0])*(bboxes[:, 3] - bboxes[:, 1])
        peaks = np.asarray([np.max(ci) for ci in blob_crops])
        counts = np.asarray([np.count_nonzero(ci) for ci in blob_crops])
        defect_part_grayscale = np.asarray([np.mean(ci[np.where(ci>0)]) for ci in blob_crops]) 
        
        densities = counts/areas
        # valid_idxs = np.where(areas >= self.area_lb)[0]
        valid_idxs = np.where(
            np.all(
                np.column_stack([areas, defect_part_grayscale]) >= self.box_thr,
                axis=1
            ) 
        )[0]
        peak_count_area = np.column_stack([peaks, counts, areas])
        if not need_crop:
            return bboxes[valid_idxs], peak_count_area[valid_idxs], densities[valid_idxs], defect_part_grayscale[valid_idxs]
        return bboxes[valid_idxs], peak_count_area[valid_idxs], densities[valid_idxs], defect_part_grayscale[valid_idxs], [blob_crops[i] for i in valid_idxs]

def detect_defect(img:np.ndarray, blober:ConnectedComponetBlob, return_draw=True)->np.ndarray|Iterable[np.ndarray]:
   
    bboxs, pca, d, g, crops = blober(img=img)
    draw = None if not return_draw else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for bi, pcai, di, gi in zip(bboxs, pca, d, g):
        
        print(f"peak : {pcai[0]:3d}, {pcai[1]:6d}/{pcai[2]:6d} = {di:.3f}, L = {gi:.2f}")
        if draw is not None:
            cv2.rectangle(
                draw, (bi[1], bi[0]), (bi[3], bi[2]),
                color=(0,0,255), thickness = 1
            )
        
    if not return_draw:
        return bboxs, pca, d, g, crops

    return bboxs, pca, d, g, crops, draw