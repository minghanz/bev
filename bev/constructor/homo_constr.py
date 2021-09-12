from .homo_constr_utils import load_pts, load_T, load_spec_dict_bev, load_calib_from_file_carla, load_calib_from_file_blender, load_vps_from_file_BrnoCompSpeed

from ..calib import Calib
from ..bev import BEVWorldSpec

import os
import yaml
import copy

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

def cfg_path_from_dataset_id(dataset_name, sub_id):
    """return the path of the yaml config file based on the dataset name and sub_id. """
    id_str = str(sub_id).replace('.', '_')
    fname = "{}_{}.yaml".format(dataset_name, id_str)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(cur_dir, 'configs_bspec', fname)
    if not os.path.exists(fpath):
        id_str = str(sub_id).split('.')[0]
        fname = "{}_{}.yaml".format(dataset_name, id_str)
        fpath = os.path.join(cur_dir, 'configs_bspec', fname)
        assert os.path.exists(fpath), "Not exist: {}".format(fpath)
    return fpath

def load_bspec_from_cfg(cfg, calib=None):
    """load bspec from a cfg dict (probably loaded from yaml). """
    mode = cfg['mode']
    spec_dict = copy.deepcopy(cfg['spec'])

    if 'm_per_px' in spec_dict:
        img_size = spec_dict['u_size'] if 'x' in spec_dict['u_axis'] else spec_dict['v_size']
        spec_dict['x_size'] = img_size * spec_dict['m_per_px']

        img_size = spec_dict['u_size'] if 'y' in spec_dict['u_axis'] else spec_dict['v_size']
        spec_dict['y_size'] = img_size * spec_dict['m_per_px']
        del spec_dict['m_per_px']

    if mode == 'abs':
        pass

    elif mode == 'offset':
        assert calib is not None, "when using `offset` mode, \
            `calib` must be given to calculate the world coordinate of the center of the original view image"
        center = calib.gen_center_in_world()

        assert ('x_max_off' in spec_dict or 'x_min_off' in spec_dict), "when using `offset` mode, \
            either `x_min_off` or `x_max_off` must be given. "

        assert ('y_max_off' in spec_dict or 'y_min_off' in spec_dict), "when using `offset` mode, \
            either `y_min_off` or `y_max_off` must be given. "

        if 'x_min_off' in spec_dict:
            spec_dict['x_min'] = center[0] + spec_dict['x_min_off']
            del spec_dict['x_min_off']
        if 'x_max_off' in spec_dict:
            spec_dict['x_max'] = center[0] + spec_dict['x_max_off']
            del spec_dict['x_max_off']
        if 'y_min_off' in spec_dict:
            spec_dict['y_min'] = center[1] + spec_dict['y_min_off']
            del spec_dict['y_min_off']
        if 'y_max_off' in spec_dict:
            spec_dict['y_max'] = center[1] + spec_dict['y_max_off']
            del spec_dict['y_max_off']

    elif mode =='centered':
        assert calib is not None, "when using `centered` mode, \
            `calib` must be given to calculate the world coordinate of the center of the original view image"
        center = calib.gen_center_in_world()

        spec_dict['x_min'] = center[0] - spec_dict['x_size'] * 0.5
        spec_dict['y_min'] = center[1] - spec_dict['y_size'] * 0.5
    else:
        raise ValueError("mode {} not recognized".format(mode))

    bspec = BEVWorldSpec(**spec_dict)
    return bspec

def load_bspec(dataset_name, sub_id=None, calib=None):
    """load bspec using yaml config files, instead of hard-coding (preset_bspec). """
    cfg_path = cfg_path_from_dataset_id(dataset_name, sub_id)
    with open(cfg_path) as f:
        cfg = yaml.load(f)
    
    bspec = load_bspec_from_cfg(cfg, calib)
    return bspec