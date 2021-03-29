import cv2
import numpy as np
from ..rbox import xywhr2xyxy, yaw2v, xywhr2xyvec, xy82xyvec, rbox_zt2tt_world
from ..converter.rbox_cvt import coco2bev, coco2bev_rboxtt

def vis_rbox(img, rboxes, mode="bev", H=None, txts=None, rbox_color=None, txt_color=None, rbox_fill=False, K=None, Rt=None, rbox_type=None):
    """rboxes is n*5 array. 
    rboxes should be of actual value (unnormalized).
    mode is "bev" or "world", corresponding to rboxes.
    H is optional if the rboxes is not in the same space as img (rboxes can be in "bev" or "world", img cam be in "bev" or "cam")
    txts is optional, if given, should be a list of strings, len(txts) == rboxes.shape[0]
    rbox_color is optional, should be a tuple or list of length 3 between 0 and 255.
    K and Rt are needed only when given xywhrt rboxes"""

    img_drawon = img.copy()
    if len(rboxes)==0:
        return img_drawon
        
    assert rboxes[:,:4].max() > 2, "Please unnormalize rboxes before visualization"
    assert mode in ["bev", "world"]

    if rbox_color is None:
        rbox_color = (0, 255, 0) # g
    if txt_color is None:
        txt_color = (255, 255, 255) # r

    if rboxes.shape[-1] == 5:
        rbox_type = "xywhr"
    elif rboxes.shape[-1] == 8:
        rbox_type = "xy8"
    elif rboxes.shape[-1] == 7:
        assert rbox_type in ["xywhrzt", "xywhrtt"]
    else:
        raise ValueError("rboxes shape not recognized: {}".format(rboxes.shape))


    ### get coordinate of vertices
    if rbox_type == "xywhr":
        xys = xywhr2xyxy(rboxes, mode)
    elif rbox_type == "xy8":
        xys = rboxes
    elif rbox_type == "xywhrzt":
        xys = xywhr2xyxy(rboxes[:,:5], mode)
        rboxtt = rbox_zt2tt_world(rboxes, K, Rt)
        xy_tails = rboxtt[:,:2] + rboxtt[:,5:]
    elif rbox_type == "xywhrtt":
        xys = xywhr2xyxy(rboxes[:,:5], mode)
        xy_tails = rboxes[:,:2] + rboxes[:,5:]
    else:
        raise ValueError("rbox_type not recognized: {}".format(rbox_type))

    xys = xys.reshape(-1, 4, 2)
    if H is not None:
        xys = cv2.perspectiveTransform(xys, H)
    xys = xys.round().astype(int)

    ### get coordinate of tails
    if rbox_type in ["xywhrt", "xywhrtt"]:
        xy_tails = xy_tails[:,None]    # expand dim at 1-dim: N*1*2
        if H is not None:
            xy_tails = cv2.perspectiveTransform(xy_tails, H)
        xy_tails = xy_tails.round().astype(int)

    ### get coordinate of directional vector
    if rbox_type == "xywhr":
        xyvec = xywhr2xyvec(rboxes, mode)
    elif rbox_type in ["xywhrt", "xywhrtt"]:
        xyvec = xywhr2xyvec(rboxes[:,:5], mode)
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
    if rbox_fill:
        img_blank = img.copy()
        cv2.fillPoly(img_blank, xys, rbox_color)
        img_drawon = cv2.addWeighted(img_drawon,0.7,img_blank,0.3,0)
    
    ### draw tails
    if rbox_type in ["xywhrt", "xywhrtt"]:
        xy_tailvec = np.concatenate([xyvec[:,[0]], xy_tails,], axis=1)
        cv2.polylines(img_drawon, xy_tailvec, False, rbox_color) #(0,255,255)

    # ### draw vec
    # cv2.polylines(img_drawon, xyvec, False, rbox_color) #(0,255,255)

    # n_obj = rboxes.shape[0]
    # for i in range(n_obj):
    #     cv2.putText(img_drawon, "x y=%.2f %.2f"%(xyvec_float[i,0,0], xyvec_float[i,0,1]), (x0[i],y0[i]), 0, 0.4, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    ### draw txts
    if txts is not None:
        assert len(txts) == rboxes.shape[0]
        n_obj = rboxes.shape[0]
        for i in range(n_obj):
            cv2.putText(img_drawon, txts[i], (x0[i], y0[i]), 0, 0.7, txt_color, thickness=1, lineType=cv2.LINE_AA)
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

def vis_anno_coco_rboxtt(img, anno_coco, width, height):
    """draw rboxtt in coco style on bev image. anno_coco is n*8 ndarray"""
# def draw_from_anno_yolo(img, anno_yolo, width, height):
    if len(anno_coco) > 0:
        rboxtts = coco2bev_rboxtt(anno_coco, width, height)
        img = vis_rbox(img, rboxtts, rbox_type="xywhrtt")
    else:
        img = img.copy()
    return img
