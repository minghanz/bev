import numpy as np
from .utils import read_txt_to_dict

def read_txt_blender_H(txt_source):
    result = read_txt_to_dict(txt_source)

    RT = result["cam_pos_inv"].reshape(4,4)
    K = result["K"].reshape(3,4)

    H_img_world = K.dot(RT)[:,[0,1,3]]

    return H_img_world

# def read_txt_blender_to_kpts(txt_source)