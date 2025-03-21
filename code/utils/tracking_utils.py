import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def update_ids(data1, data2):
    num_boxes1 = len(data1.boxes)
    num_boxes2 = len(data2.boxes)
    
    cost_matrix = np.zeros((num_boxes1, num_boxes2))
    iou_thresholds = np.zeros((num_boxes1, num_boxes2))
    
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            area1 = (data1.boxes[i][2] - data1.boxes[i][0]) * (data1.boxes[i][3] - data1.boxes[i][1])
            area2 = (data2.boxes[j][2] - data2.boxes[j][0]) * (data2.boxes[j][3] - data2.boxes[j][1])
            avg_area = (area1 + area2) / 2
            iou_threshold = avg_area * 0.001  # Dynamic threshold based on area
            iou_thresholds[i, j] = iou_threshold
            
            cost_matrix[i, j] = 1 - iou(data1.boxes[i], data2.boxes[j])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assigned = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= iou_thresholds[r, c]:  # Dynamic IoU threshold
            data2.ids[c] = data1.ids[r]
            assigned.add(c)
    
    for i in range(num_boxes2):
        if i not in assigned:
            data2.ids[i] = data1.max_id + 1
            data1.max_id += 1  # Increment max_id for new objects
    if len(data2.ids) > 0:
        data2.max_id = max(data2.ids)
    else:
        data2.max_id = None