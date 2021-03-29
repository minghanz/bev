import numpy as np
from ..io import *
from ..visualizer import *


def draw_homography_from_blender_txt(img, txt_path):
    H_img_world = homo_io.read_txt_blender_H(txt_path)

    xn = 2
    yn = 2
    grid_x = np.array(list(range(-xn, xn+1))) * 4 + 2
    grid_y = np.array(list(range(-yn, yn+1))) *8 + 10
    
    img = homo_vis.draw_homography(img, H_img_world, grid_x, grid_y)

    return img

def draw_kpts_from_blender_txt(img, txt_path, H=None):
    ### draw detections
    kpts = kpts_io.read_txt_blender_kpts(txt_path)
    img = kpts_vis.vis_kpts(img, kpts, H)

    return img

def draw_from_blender_txt(img, txt_path, H=None):
    
    """can work with both gt file and prediction, since both have 'bbox_#' and 'pts_proj' terms. """
    result = read_txt_to_dict(txt_path)

    ### if there are camera intr/extr info, draw the estimated road plane by a grid
    if "K" in result and "cam_pos_inv" in result:
        img = draw_homography_from_blender_txt(img, txt_path)

    if any("pts_proj" in x for x in result):
        img = draw_kpts_from_blender_txt(img, txt_path, H)
    
    # if any("bbox" in x for x in result):
    #     img = draw_bbox(img, result)

    return img

if __name__ == "__main__":
    pass