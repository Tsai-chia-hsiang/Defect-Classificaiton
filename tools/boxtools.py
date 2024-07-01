from typing import Iterable
from collections import deque
import numpy as np
import torch
from torchvision.ops import box_iou
import cv2

def crop(img:np.ndarray, xyxy:np.ndarray)->np.ndarray:
   
    return img[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3], ...].copy()

def compute_boundary_distance(box1, box2):
    """
    Compute the minimum boundary distance between two bounding boxes.
    
    Parameters:
    box1, box2: list or tuple of (x1, y1, x2, y2)
        (x1, y1) is the top-left coordinate
        (x2, y2) is the bottom-right coordinate
    
    Returns:
    float: Minimum boundary distance
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    # Compute horizontal and vertical distances
    horizontal_dist = max(left2 - right1, left1 - right2, 0)
    vertical_dist = max(top2 - bottom1, top1 - bottom2, 0)
    
    # If boxes overlap, distance is zero
    if horizontal_dist == 0 and vertical_dist == 0:
        return 0
    
    # Otherwise, return the Euclidean distance
    return np.sqrt(horizontal_dist ** 2 + vertical_dist ** 2)

def pairwise_is_box_inside(box1:torch.Tensor, box2:torch.Tensor)->torch.Tensor:
    """
    Check if each `inner_box` is completely inside any `outer_box`.
    
    Parameters:
    outer_boxes (Tensor): Tensor of shape (N, 4) representing N outer bounding boxes [x1, y1, x2, y2]
    inner_boxes (Tensor): Tensor of shape (M, 4) representing M inner bounding boxes [x1, y1, x2, y2]
    
    Returns:
    Tensor: A boolean tensor of shape (N, M) where each element (i, j) is True if `inner_boxes[j]` is inside `outer_boxes[i]`
    """
    outer_boxes = box1.unsqueeze(1)  # Shape (N, 1, 4)
    inner_boxes = box2.unsqueeze(0)  # Shape (1, M, 4)
    
    inside_x1 = inner_boxes[..., 0] >= outer_boxes[..., 0]
    inside_y1 = inner_boxes[..., 1] >= outer_boxes[..., 1]
    inside_x2 = inner_boxes[..., 2] <= outer_boxes[..., 2]
    inside_y2 = inner_boxes[..., 3] <= outer_boxes[..., 3]
    
    return (inside_x1 & inside_y1 & inside_x2 & inside_y2).to(torch.int32)



def merge_boxes(box1, box2):
    """
    Merge two bounding boxes.
    
    Parameters:
    box1, box2: list or tuple of (x1, y1, x2, y2)
    
    Returns:
    list: Merged bounding box
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    
    return [x1, y1, x2, y2]

def merge_boxes_v(boxes:np.ndarray) -> np.ndarray:
    """
    Args
    -------
    boxes: (N, 4) : (N, xyxy)

    Return
    ------
    a merged box xyxy
    """
    xmin = np.min(boxes[:, 0])
    ymin = np.min(boxes[:, 1])
    xmax = np.max(boxes[:, 2])
    ymax = np.max(boxes[:, 3])
    return np.array([xmin, ymin, xmax, ymax])

def merge_boxes_with_iou2(boxes:np.ndarray, d:float=10):
    """
    boxes: (N, 4)
    """
    def grouping(src_boxes:np.ndarray, cost_matrix:np.ndarray, cmp_funct) -> np.ndarray:
        
        still_exist = (np.arange(src_boxes.shape[0])).tolist()
        group = []
        
        while still_exist:

            target = still_exist[0]
            still_exist = still_exist[1:]
            overlap = np.where(cmp_funct(cost_matrix[target]))[0]
            
            gi = [target]
            for fi in overlap:
                if fi in still_exist:
                    gi.append(fi)
                    still_exist.remove(fi)
            gi = np.asarray(gi)
            group.append(merge_boxes_v(src_boxes[gi]))
        
        return np.asarray(group)

    # first merge : merge fully covered 
    merge = boxes

    for c in range(10):
        box_tensor = torch.from_numpy(merge)
        
        pairwised_covered = pairwise_is_box_inside(box_tensor, box_tensor).numpy()
        pairwise_iou = box_iou(box_tensor, box_tensor).numpy()
    
        if np.sum(pairwised_covered) == merge.shape[0] and np.sum(pairwise_iou) == 1.0*merge.shape[0]:
            # print(c)
            break
        
        merge = grouping(
            src_boxes=merge, 
            cost_matrix=pairwised_covered, 
            cmp_funct = lambda x:x==1
        )
        
        # second : merge iou > 0
        box_tensor = torch.from_numpy(merge)
        pairwise_iou = box_iou(box_tensor, box_tensor).numpy()
        merge = grouping(
            src_boxes=merge, 
            cost_matrix=pairwise_iou, 
            cmp_funct = lambda x:x>0
        )
    # third merge with boundary distance and then iou merge again
    
    merge = np.asarray(
        merge_boxes_with_boundary_distance(merge.tolist(), d)
    )
    box_tensor = torch.from_numpy(merge)
    pairwise_iou = box_iou(box_tensor, box_tensor).numpy()
    merge = grouping(
            src_boxes=merge, 
            cost_matrix=pairwise_iou, 
            cmp_funct = lambda x:x>0
        )
    return merge 

def merge_boxes_with_boundary_distance(boxes, threshold_distance):
    """
    Merge a list of bounding boxes based on boundary distance.
    
    Parameters:
    boxes: list of lists or tuples of (x1, y1, x2, y2)
    threshold_distance: float
    
    Returns:
    list: Merged bounding boxes
    """
    merged_boxes = []

    while boxes:
        base_box = boxes.pop(0)
        to_merge = []
        
        for box in boxes:
            if compute_boundary_distance(base_box, box) <= threshold_distance:
                to_merge.append(box)
        
        for box in to_merge:
            boxes.remove(box)
            base_box = merge_boxes(base_box, box)
        
        merged_boxes.append(base_box)
    
    return merged_boxes

def non_max_suppression_fast(boxes:np.ndarray, overlapThresh:float=0.3) -> list:
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

def is_bbox_at_edge_or_corner(bbox, image_shape, thr = 10) -> bool:

    x_min, y_min, x_max, y_max = bbox
    width, height  = image_shape[:2]
    
    left_margin = x_min
    right_margin = width - x_max
    top_margin = y_min
    bottom_margin = height - y_max
    #print(bbox)
    #print(left_margin, top_margin, right_margin,  bottom_margin)
    #print(right_margin, thr, right_margin < thr)
    W = (left_margin <= thr) or (right_margin <= thr)
    #print(W)
    H = (top_margin <= thr) or (bottom_margin <= thr)
    #print(H)
    return (W or H)

def print_box(box:dict, bid=None)->str:
    f = "" if bid is None else f"{bid}\n"
    l = len(box)
    count = 0
    for k,v in box.items():
        if k == 'lines':
            f+= f"    N:{len(v[0])}\n    line_light:{v[1]}\n    line_den:{v[2]}\n    line_length:{v[3]}"
        else:
            f+= f"    {k}:{v}"
        if count < l-1:
            f+="\n"
        count += 1
    return f
