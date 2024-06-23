from typing import Literal, Any
import cv2
import numpy as np

def non_max_suppression_fast(boxes:np.ndarray, overlapThresh:float=0.3) -> np.ndarray:
    if len(boxes) == 0:
        return []
            
    # Initialize the list of picked indexes 
    pick = []
    
    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    # Return only the bounding boxes that were picked
    return boxes[pick].astype(np.int32)

def crop(img:np.ndarray, xyxy:np.ndarray)->np.ndarray:
   
    return img[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3], ...]

class ConnectedComponetBlob():

    def __init__(self, pixel_value_thr:int=0, blob_area_lowerbound:int=200, peak_lowerbound:int=220):
        
        self.pixel_thr = pixel_value_thr

        # [area, peak]
        self.boxes_thr = np.array([blob_area_lowerbound, peak_lowerbound],dtype=np.float32)
    
    def __call__(self, img:np.ndarray, return_type:Literal["mask", "bbox"]="bbox", nms:bool=True, input_color_mode:Literal["RGB","BGR"]="BGR") -> tuple[int, np.ndarray] | list[dict[str, Any]]:
            
        binary_image = (
            ( (img > self.pixel_thr).astype(np.int32) )*255
        ).astype(np.uint8)
        
        num_labels, labels_im = cv2.connectedComponents(binary_image)
        if return_type == "mask":
            return num_labels, labels_im
        
        boxes = self.get_bbox_from_mask(num_labels=num_labels, mask=labels_im, nms=nms)
        # return [{'box':i} for i in boxes]
        return self._bbox_post_processing(img=img, bboxes=boxes)
    
    def get_bbox_from_mask(self, num_labels:int, mask:np.ndarray, nms:bool=True) -> np.ndarray:
        
        bounding_boxes = []
        # Start from 1 to ignore the background
        for label in range(1, num_labels):  
            component_mask = (mask == label).astype(np.uint8)
            coords = np.column_stack(np.where(component_mask > 0))
            if coords.size > 0:
                x, y, w, h = cv2.boundingRect(coords)
                # Store as (x1, y1, x2, y2)
                bounding_boxes.append([x, y, x + w, y + h])  
        bounding_boxes = np.asarray(bounding_boxes)
        
        return bounding_boxes if not nms \
            else non_max_suppression_fast(bounding_boxes)

    def _bbox_post_processing(self, img:np.ndarray, bboxes:np.ndarray)->list[dict[str, Any]]:
        if len(bboxes) == 0:
            return []
        
        blob_crops = [crop(img=img, xyxy=bi) for bi in bboxes]

        areas = (bboxes[:, 2]- bboxes[: ,0])*(bboxes[:, 3] - bboxes[:, 1])
        peaks = np.array([np.max(ci) for ci in blob_crops])
        densities = np.array([np.count_nonzero(ci)/ci.size for ci in blob_crops])
        
        condition = np.column_stack([areas, peaks])

        comparison_result = condition >= self.boxes_thr
        valid_idxs = np.where(np.all(comparison_result, axis=1))[0]

        return [
            {
                'box':bboxes[i],
                'area':areas[i],
                'peak':peaks[i],
                'density':densities[i]
            } 
            for i in valid_idxs
        ]