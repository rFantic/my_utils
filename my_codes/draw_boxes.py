import cv2
import numpy as np

def draw_box_xyxy(img, box_xyxy, color=(255,0,0), size=2, inplace=False, text=None):
    if not inplace: img = img.copy()
    if text:
        x, y = tuple(box_xyxy[:2])
        cv2.putText(img, text, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,255), 2)
    return cv2.rectangle(img, tuple(box_xyxy[:2]), tuple(box_xyxy[2:]), color, size)
    
def draw_box_xywh(img, box_xywh, color=(255,0,0), size=2, inplace=False, text=None):
    if not inplace: img = img.copy()
    if text:
        x, y = tuple(box_xywh[:2])
        cv2.putText(img, text, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,255), 2)
    return cv2.rectangle(img, tuple(box_xywh[:2]), 
                  (box_xywh[0]+box_xywh[2], box_xywh[1] + box_xywh[3]), 
                  color, size)

def draw_boxes_xyxy(img, boxes_xyxy, color=(255,0,0), size=2, enable_label=False):
    img = img.copy()
    for i, box_xyxy in enumerate(boxes_xyxy):
        draw_box_xyxy(img, box_xyxy, color, size, True, text=None)#'{}'.format(i))
    return img
    
def draw_boxes_xywh(img, boxes_xywh, color=(255,0,0), size=2, enable_label=False):
    img = img.copy()
    for i, box_xywh in enumerate(boxes_xywh):
        draw_box_xywh(img, box_xywh, color, size, True, text=None)#'{}'.format(i))
    return img


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')
