from .homo_constr_utils import load_pts, load_T, load_spec_dict_bev, load_calib_from_file_carla, load_calib_from_file_blender, load_vps_from_file_BrnoCompSpeed

from ..calib import Calib
from ..bev import BEVWorldSpec

def preset_calib(dataset_name, sub_id=None):
    assert dataset_name in ["lturn", "KoPER", "roundabout"]
    if dataset_name in ["lturn", "roundabout"]:
        width = 852
        height = 480
        pts_3d, pts_2d = load_pts(dataset_name, width, height, sub_id)

        calib = Calib(pts_world=pts_3d, pts_image=pts_2d, u_size=width, v_size=height)

    elif dataset_name == "KoPER":
        assert sub_id in [1, 4]
        width = 656
        height = 494
        fx, fy, cx, cy, T = load_T(dataset_name, sub_id)

        calib = Calib(fx=fx, fy=fy, cx=cx, cy=cy, T=T, u_size=width, v_size=height)

    return calib

def load_calib(dataset_name, fpath):
    if dataset_name == "CARLA":
        K, T_cam_world, u_size, v_size = load_calib_from_file_carla(fpath)
        calib = Calib(K=K, T=T_cam_world, u_size=int(u_size), v_size=int(v_size))
    elif dataset_name == "blender":
        K, T_cam_world, u_size, v_size = load_calib_from_file_blender(fpath)
        calib = Calib(K=K, T=T_cam_world, u_size=int(u_size), v_size=int(v_size))
    elif dataset_name == "BrnoCompSpeed":
        width = 1920
        height = 1080
        cal = load_vps_from_file_BrnoCompSpeed(fpath)
        calib = Calib(vp1=cal["vp1"], vp2=cal["vp2"], pp=cal["pp"], height=cal["height"], u_size=width, v_size=height)
    else:
        raise ValueError("dataset_name {} not recognized", dataset_name)
    return calib


def preset_bspec(dataset_name, sub_id=None, calib=None):
    assert dataset_name in ["lturn", "KoPER", "kitti", "CARLA", "roundabout", "BrnoCompSpeed", "rounD", "rounD_raw"]

    if dataset_name == "lturn":
        #### world to bev homography
        # bev_mask = cv2.imread("/home/minghanz/Pictures/empty_road/bev/road_mask_bev.png")
        # bev_width = bev_mask.shape[1] #416
        # bev_height = bev_mask.shape[0] #544#480 #544 # 624
        if sub_id == 0:
            bev_width = 416
            # bev_width = 384
            bev_height = 544
        else:
            bev_width = 416
            bev_height = 672

        spec_dict = load_spec_dict_bev(bev_width, bev_height, dataset_name, sub_id)
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

    elif dataset_name == "kitti":
        bev_width = 224#192#384
        bev_height = 544#512#576
        # bspec = bev.bev.BEVWorldSpec(u_size=400, v_size=800, u_axis="x", v_axis="-y", x_min=-10, x_max=10, y_min=6, y_max=46)
        # bspec = BEVWorldSpec(u_size=bev_width, v_size=bev_height, u_axis="x", v_axis="-y", x_min=-20, x_max=20, y_min=6, y_max=66)
        # bspec = BEVWorldSpec(u_size=bev_width, v_size=bev_height, u_axis="x", v_axis="-y", x_min=-15, x_max=15, y_min=6, y_max=86)
        bspec = BEVWorldSpec(u_size=bev_width, v_size=bev_height, u_axis="x", v_axis="-y", x_min=-14, x_max=14, y_min=6, y_max=74)

    elif dataset_name == "CARLA":
        # assert sub_id in [1, 2, 3, 4]
        bev_width = 544
        bev_height = 544

        spec_dict = load_spec_dict_bev(bev_width, bev_height, dataset_name, sub_id)
        bspec = BEVWorldSpec(**spec_dict)
        
    elif dataset_name == "roundabout":
        #### world to bev homography
        # bev_mask = cv2.imread("/home/minghanz/Pictures/empty_road/bev/road_mask_bev.png")
        # bev_width = bev_mask.shape[1] #416
        # bev_height = bev_mask.shape[0] #544#480 #544 # 624
        bev_width = 544
        bev_height = 544

        spec_dict = load_spec_dict_bev(bev_width, bev_height, dataset_name, sub_id)
        bspec = BEVWorldSpec(**spec_dict)

    elif dataset_name == "rounD" or dataset_name == "rounD_raw":
        if sub_id == 0:
            bev_width = 1544
            bev_height = 936
        elif sub_id == 1:
            bev_width = 1678
            bev_height = 936
        elif sub_id >= 2:
            bev_width = 1678
            bev_height = 936
        else:
            raise ValueError("sub_id {} not recognized. ".format(sub_id))

        spec_dict = load_spec_dict_bev(bev_width, bev_height, dataset_name, sub_id)
        bspec = BEVWorldSpec(**spec_dict)

    elif dataset_name == "BrnoCompSpeed":
        if sub_id == 0:
            bev_width = 384
            bev_height = 768       # 32px increment
        elif sub_id == 4.1:
            bev_width = 256
            bev_height = 416       # 32px increment
        elif sub_id == 4.2:
            bev_width = 256
            bev_height = 384
        elif sub_id == 4.3:
            bev_width = 224
            bev_height = 512
        elif sub_id == 5.1:
            bev_width = 288
            bev_height = 448       # 32px increment
        elif sub_id == 5.2:
            bev_width = 192
            bev_height = 320
        elif sub_id == 5.3:
            bev_width = 192
            bev_height = 448
        elif sub_id == 6.1:
            bev_width = 320
            bev_height = 640       # 32px increment
        elif sub_id == 6.2:
            bev_width = 288
            bev_height = 640       # 32px increment
        elif sub_id == 6.3:
            bev_width = 192
            bev_height = 512       # 32px increment
            
        spec_dict = load_spec_dict_bev(bev_width, bev_height, dataset_name, sub_id, calib=calib)
        bspec = BEVWorldSpec(**spec_dict)

    return bspec

