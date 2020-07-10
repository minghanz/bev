from .homo_constr_utils import load_pts, load_T, load_spec_dict_bev

from ..calib import Calib
from ..bev import BEVWorldSpec

def preset_calib(dataset_name, sub_id=None):
    assert dataset_name in ["lturn", "KoPER"]
    if dataset_name == "lturn":
        width = 852
        height = 480
        pts_3d, pts_2d = load_pts(dataset_name, width, height)

        calib = Calib(pts_world=pts_3d, pts_image=pts_2d, u_size=width, v_size=height)

    elif dataset_name == "KoPER":
        assert sub_id in [1, 4]
        width = 656
        height = 494
        fx, fy, cx, cy, T = load_T(dataset_name, sub_id)

        calib = Calib(fx=fx, fy=fy, cx=cx, cy=cy, T=T, u_size=width, v_size=height)

    return calib

def preset_bspec(dataset_name, sub_id=None):
    assert dataset_name in ["lturn", "KoPER"]

    if dataset_name == "lturn":
        #### world to bev homography
        # bev_mask = cv2.imread("/home/minghanz/Pictures/empty_road/bev/road_mask_bev.png")
        # bev_width = bev_mask.shape[1] #416
        # bev_height = bev_mask.shape[0] #544#480 #544 # 624
        bev_width = 416
        bev_height = 544

        spec_dict = load_spec_dict_bev(bev_width, bev_height, dataset_name)
        bspec = BEVWorldSpec(**spec_dict)

    elif dataset_name == "KoPER":
        assert sub_id in [1, 4]
        # bev_mask = cv2.imread("/media/sda1/datasets/extracted/KoPER/added/SK_{}_empty_road_mask_bev.png".format(sub_id))
        # bev_width = bev_mask.shape[1] #416
        # bev_height = bev_mask.shape[0] #544#480 #544 # 624
        bev_width = 544
        bev_height = 416

        spec_dict = load_spec_dict_bev(bev_width, bev_height, dataset_name, sub_id)
        bspec = BEVWorldSpec(**spec_dict)

    return bspec

