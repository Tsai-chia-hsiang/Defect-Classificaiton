from typing import Literal, Any, Iterable
import cv2
import gc
import numpy as np
from .boxtools import *

class ConnectedComponetBlob():

    def __init__(
        self, pixel_value_thr:int=0, 
        min_pixels_per_comp:int=20, 
        blob_area_lowerbound:int=80, 
        region_mean:float=30,
        box_gray_lowerbound:float=25,
        box_merge_distance:float=10
    ):
        
        self.pixel_thr = pixel_value_thr
        self.componet_lb = min_pixels_per_comp
        self.rmean = region_mean
        self.merge_dist = box_merge_distance
        self.area_lb = blob_area_lowerbound
        self.box_gray_lb = box_gray_lowerbound
        self.box_thr = np.array([self.area_lb, box_gray_lowerbound])
    
    def __call__(self, img:np.ndarray, need_crop:bool=True, topk_light:int=0) -> list[dict[str, Any]]|tuple[list[dict[str, Any]], list[np.ndarray]]:

        
        binary_image = (
            ( (img > self.pixel_thr).astype(np.int32) )*255
        ).astype(np.uint8)
        
        num_labels, labels_im = cv2.connectedComponents(binary_image)
  
        boxes = self.get_bbox_from_mask(
            img=img, num_labels=num_labels, mask=labels_im
        )
        
        return self.bbox_post_processing(
            img=img, bboxes=boxes, 
            need_crop=need_crop, topk=topk_light
        )
   
    
    def get_bbox_from_mask(self, img:np.ndarray, num_labels:int, mask:np.ndarray) -> np.ndarray: 
        
        
        bounding_boxes = []
        # Start from 1 to ignore the background
        for label in range(1, num_labels):  
            component_mask = (mask == label).astype(np.uint8)
            coords = np.column_stack(np.where(component_mask > 0))
            if coords.size/2 >= self.componet_lb:  
                m = np.mean(img[np.where(component_mask > 0)].astype(np.float32))
                if m >= self.rmean:  
                    x, y, w, h = cv2.boundingRect(coords)
                    bounding_boxes.append([x, y, x + w, y + h])  
        
        bounding_boxes = np.asarray(bounding_boxes)
        """
        bounding_boxes = non_max_suppression_fast(np.asarray(bounding_boxes))
        """
        
        areas = (bounding_boxes[:, 2]- bounding_boxes[: ,0])*\
            (bounding_boxes[:, 3] - bounding_boxes[:, 1])
        
        bounding_boxes = bounding_boxes[np.argsort(-areas)]
        M = merge_boxes_with_iou(bounding_boxes, d=self.merge_dist)

        return M
    
    def bbox_post_processing(self, img:np.ndarray, bboxes:np.ndarray, need_crop:bool, topk:int=0)->list[dict[str, Any]]|tuple[list[dict[str, Any]], list[np.ndarray]]:
        
        if len(bboxes) == 0:
            return [], [] if need_crop else []
        
        blob_crops = [crop(img=img, xyxy=bi) for bi in bboxes]
     
        areas = (bboxes[:, 2]- bboxes[: ,0])*(bboxes[:, 3] - bboxes[:, 1])
        peaks = np.asarray([np.max(ci) for ci in blob_crops])
        counts = np.asarray([np.count_nonzero(ci) for ci in blob_crops])
        Lcounts = np.asarray([np.count_nonzero(ci > 80) for ci in blob_crops])
        defect_part_grayscale = np.asarray([np.mean(ci[np.where(ci>0)]) for ci in blob_crops]) 
        
        densities = counts/areas
        
        valid_idxs = np.where(
            np.all(
                np.column_stack([areas, defect_part_grayscale]) >= self.box_thr,
                axis=1
            ) 
        )[0]

        sorted_idxs = np.argsort(-defect_part_grayscale[valid_idxs])
        valid_idxs = valid_idxs[sorted_idxs]
       
        if topk > 0:
            valid_idxs = valid_idxs[:topk]
            
        bboxs_des = [
            {
                'xyxy'  : bboxes[i],
                'ncount' : counts[i],
                'lcount' : Lcounts[i],
                'lr'     : Lcounts[i] / counts[i],
                'peak' : peaks[i],
                'area'  : areas[i],
                'density': densities[i] ,
                'corner': is_bbox_at_edge_or_corner(
                    bbox=bboxes[i], 
                    image_shape=img.shape, 
                    thr=0
                ),
                'avg_light':defect_part_grayscale[i]
            }   
            for i in valid_idxs
        ]
        
        
        del areas ,peaks, counts ,Lcounts ,defect_part_grayscale, densities
        
        
        if not need_crop:
            del blob_crops
            gc.collect()
            return bboxs_des
        else:
            gc.collect()
            return bboxs_des, [blob_crops[i] for i in valid_idxs]
        