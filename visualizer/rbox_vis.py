import cv2
import numpy as np
from ..rbox import xywhr2xyxy, yaw2v, xywhr2xyvec, xy82xyvec
from ..converter.rbox_cvt import coco2bev

def vis_rbox(img, rboxes, mode="bev", H=None, txts=None, rbox_color=None, txt_color=None):
    """rboxes is n*5 array. 
    rboxes should be of actual value (unnormalized).
    mode is "bev" or "world", corresponding to rboxes.
    H is optional if the image is not in the same space as img 
    txts is optional, if given, should be a list of strings, len(txts) == rboxes.shape[0]
    rbox_color is optional, should be a tuple or list of length 3 between 0 and 255 """

    assert rboxes[:,:4].max() > 2, "Please unnormalize rboxes before visualization"
    assert mode in ["bev", "world"]

    if rbox_color is None:
        rbox_color = (0, 255, 0) # g
    if txt_color is None:
        txt_color = (0, 0, 255) # r

    if rboxes.shape[-1] == 5:
        rbox_type = "xywhr"
    elif rboxes.shape[-1] == 8:
        rbox_type = "xy8"
    else:
        raise ValueError("rboxes shape not recognized: {}".format(rboxes.shape))

    img_drawon = img.copy()

    ### get coordinate of vertices
    if rbox_type == "xywhr":
        xys = xywhr2xyxy(rboxes, mode)
    else:
        xys = rboxes

    xys = xys.reshape(-1, 4, 2)
    if H is not None:
        xys = cv2.perspectiveTransform(xys, H)
    xys = xys.round().astype(int)

    ### get coordinate of directional vector
    if rbox_type == "xywhr":
        xyvec = xywhr2xyvec(rboxes, mode)
    else:
        xyvec = xy82xyvec(rboxes)

    xyvec = xyvec.reshape(-1, 2, 2)
    if H is not None:
        xyvec = cv2.perspectiveTransform(xyvec, H)
    xyvec_float = xyvec.copy()
    xyvec = xyvec.round().astype(int)
    x0 = xyvec[:,0,0]
    y0 = xyvec[:,0,1]
    x1 = xyvec[:,1,0]
    y1 = xyvec[:,1,1]

    ### draw box
    cv2.polylines(img_drawon, xys, True, rbox_color) # (0,0,255)

    ### draw vec
    cv2.polylines(img_drawon, xyvec, False, rbox_color) #(0,255,255)

    # n_obj = rboxes.shape[0]
    # for i in range(n_obj):
    #     cv2.putText(img_drawon, "x y=%.2f %.2f"%(xyvec_float[i,0,0], xyvec_float[i,0,1]), (x0[i],y0[i]), 0, 0.4, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    ### draw txts
    if txts is not None:
        assert len(txts) == rboxes.shape[0]
        n_obj = rboxes.shape[0]
        for i in range(n_obj):
            cv2.putText(img_drawon, txts[i], (x0[i], y0[i]), 0, 0.4, txt_color, thickness=1, lineType=cv2.LINE_AA)
    return img_drawon


def vis_anno_coco(img, anno_coco, width, height):
    """draw rbox in coco style on bev image. anno_coco is n*6 ndarray"""
# def draw_from_anno_yolo(img, anno_yolo, width, height):
    if len(anno_coco) > 0:
        rboxes = coco2bev(anno_coco, width, height)
        img = vis_rbox(img, rboxes)
    else:
        img = img.copy()
    return img