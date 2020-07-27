import numpy as np
from .utils import read_txt_to_dict

def read_txt_blender_kpts(txt_source):
    result = read_txt_to_dict(txt_source)

    n_obj = len([key for key in result if "pts_proj" in key])

    kpts_img = np.zeros((n_obj, 4,3)) # n*4*2
    for i_obj in range(n_obj):
        if "pts_proj" in result:
            pts = result['pts_proj']
        else:
            pts = result['pts_proj_{}'.format(i_obj)]
        pts = pts.reshape(-1, 3)
        kpts_img[i_obj] = pts#[:, :2]
    
    return kpts_img
