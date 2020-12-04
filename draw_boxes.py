import cv2

def draw_box_xyxy(img, box_xyxy, color=(255,0,0), size=2, inplace=False):
    if not inplace: img = img.copy()
    return cv2.rectangle(img, tuple(box_xyxy[:2]), tuple(box_xyxy[2:]), color, size)
    
def draw_box_xywh(img, box_xywh, color=(255,0,0), size=2, inplace=False):
    if not inplace: img = img.copy()
    return cv2.rectangle(img, tuple(box_xywh[:2]), 
                  (box_xywh[0]+box_xywh[2], box_xywh[1] + box_xywh[3]), 
                  color, size)

def draw_boxes_xyxy(img, boxes_xyxy, color=(255,0,0), size=2):
    img = img.copy()
    for box_xyxy in boxes_xyxy:
        draw_box_xyxy(img, box_xyxy, color, size, True)
    return img
    
def draw_boxes_xywh(img, boxes_xywh, color=(255,0,0), size=2):
    img = img.copy()
    for box_xywh in boxes_xywh:
        draw_box_xywh(img, box_xywh, color, size, True)
    return img
