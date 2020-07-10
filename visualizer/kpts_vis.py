import numpy as np
import cv2

def vis_kpts(img, pts, H=None):
    """visualize keypoints, which should be n*4*2 ndarray or 4*2 array"""

    assert pts.shape[-1] == 2, "pts should be n*4*2 or 4*2 ndarray, {}".format(pts.shape)

    img_drawon = img.copy()

    pts = pts.reshape(-1, 4,2)
    if H is not None:
        pts = cv2.perspectiveTransform(pts, H)
    pts = pts.round().astype(int)

    kpt_color = (0, 0, 255)
    kptline_color = (255, 0, 0)

    # ### draw cross
    # cv2.line(img, (int(round(pts[0,0])), int(round(pts[0,1]))), (int(round(pts[3,0])), int(round(pts[3,1]))), kptline_color, thickness=1)
    # cv2.line(img, (int(round(pts[1,0])), int(round(pts[1,1]))), (int(round(pts[2,0])), int(round(pts[2,1]))), kptline_color, thickness=1)

    ### draw rectangle
    cv2.polylines(img_drawon, pts[:,[0,1,3,2]], True, kptline_color, thickness=1)
    # cv2.line(img, (int(round(pts[0,0])), int(round(pts[0,1]))), (int(round(pts[1,0])), int(round(pts[1,1]))), kptline_color, thickness=1)
    # cv2.line(img, (int(round(pts[1,0])), int(round(pts[1,1]))), (int(round(pts[3,0])), int(round(pts[3,1]))), kptline_color, thickness=1)
    # cv2.line(img, (int(round(pts[3,0])), int(round(pts[3,1]))), (int(round(pts[2,0])), int(round(pts[2,1]))), kptline_color, thickness=1)
    # cv2.line(img, (int(round(pts[2,0])), int(round(pts[2,1]))), (int(round(pts[0,0])), int(round(pts[0,1]))), kptline_color, thickness=1)

    ### draw points
    pts_H_flat = pts.reshape(-1,2)
    for i in range(pts_H_flat.shape[0]):
        cv2.circle(img_drawon, (pts_H_flat[i,0], pts_H_flat[i,1]), radius=3, color=kpt_color)
    
    return img_drawon