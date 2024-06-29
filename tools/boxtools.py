from typing import Iterable
import numpy as np
import cv2

def crop(img:np.ndarray, xyxy:np.ndarray)->np.ndarray:
   
    return img[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3], ...].copy()

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1, box2: list or tuple of (x1, y1, x2, y2)
        (x1, y1) is the top-left coordinate
        (x2, y2) is the bottom-right coordinate
    
    Returns:
    float: IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x1_inter < x2_inter and y1_inter < y2_inter:
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        inter_area = 0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

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

def merge_boxes_with_iou(boxes):
    """
    Merge a list of bounding boxes where the IoU of each pair of boxes > 0.
    
    Parameters:
    boxes: list of lists or tuples of (x1, y1, x2, y2)
    
    Returns:
    list: Merged bounding boxes
    """
    merged_boxes = []

    while boxes:
        base_box = boxes.pop(0)
        to_merge = []
        
        for box in boxes:
            if compute_iou(base_box, box) > 0:
                to_merge.append(box)
        
        for box in to_merge:
            boxes.remove(box)
            base_box = merge_boxes(base_box, box)
        
        merged_boxes.append(base_box)
    
    return merged_boxes


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

def bbox_wh_ratio(bbox:np.ndarray):
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]
    if min(w,h) == 0 :
        return 0

    return min(w,h) / max(w,h)

def merge_lines(lines, angle_threshold=5, distance_threshold=20):
    
    def angle_between_lines(line1, line2):
        def get_angle(x1, y1, x2, y2):
            return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        angle1 = get_angle(x1, y1, x2, y2)
        angle2 = get_angle(x3, y3, x4, y4)
    
        return abs(angle1 - angle2)

    if lines is None:
        return []

    if len(lines) == 1:
        return lines[0]

    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        added = False
        for merged_line in merged_lines:
            mx1, my1, mx2, my2 = merged_line
            angle_diff = angle_between_lines([x1, y1, x2, y2], merged_line)
            if angle_diff < angle_threshold and ((abs(x1 - mx1) < distance_threshold and abs(y1 - my1) < distance_threshold) or \
                                                 (abs(x2 - mx2) < distance_threshold and abs(y2 - my2) < distance_threshold)):
                merged_line[0] = min(x1, mx1)
                merged_line[1] = min(y1, my1)
                merged_line[2] = max(x2, mx2)
                merged_line[3] = max(y2, my2)
                added = True
                break
        if not added:
            merged_lines.append([x1, y1, x2, y2])
    return merged_lines



def line_light_and_density(points:np.ndarray, crop:np.ndarray) -> tuple[float, float]:
    
    def get_line_points(start_point, end_point):
        # Create a black image of sufficient size
        img = np.zeros_like(crop)

        cv2.line(img, start_point, end_point, 255, 1)
        
        return np.where(img == 255)

    points_gray_scale = crop[
        get_line_points(
            (int(points[0]), int(points[1])),
            (int(points[2]), int(points[3]))
        )
    ].astype(np.int32)
    
    density = np.count_nonzero(points_gray_scale)/points_gray_scale.size
    avg_light = np.mean(points_gray_scale[np.where(points_gray_scale > 0)])
    
    return float(avg_light), density

def n_line_single(c) -> tuple[list[np.ndarray], list[float], list[float]]:
    
    edges = cv2.Canny(c, 50, 150)
    lines = merge_lines(
        cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=20, minLineLength=25, maxLineGap=10
        )
    )
    d = []
    valid_l = []
    avg_l = []
    l_length = []
    for li in lines:
        light_i, di = line_light_and_density(points=li, crop=c)
        len_i = cv2.norm((int(li[0]), int(li[1])), (int(li[2]), int(li[3])))
        # print(light_i, di)
        if light_i > 45:
            l_length.append(len_i)
            valid_l.append(li)
            d.append(di)
            avg_l.append(light_i)
    
    return valid_l, avg_l, d, l_length


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
