import numpy as np
from ..io import read_txt_to_array, read_txt_to_dict
import json

def load_pts(name, img_width, img_height, sub_id=None):

    ### for lturn, if sub_id=0, use legacy spec
    if name == "lturn":
        assert sub_id is None or sub_id == 0
        if sub_id == 0:
            ### load homographic calibration and draw on picture
            pts_3d = np.array( [[0, 0, 0], [3.7, 0, 0], [7.4, 0, 0], [-0.87, 21.73, 0],
                            [-4.26, 21.73, 0], [-5.22, -2.17, 0]], dtype=np.float32)
        else:
            ### 20201224
            pts_3d_ = np.array([[0, 0, 0],
                            [-0.143540669856454, -3.44497607655502, 0],
                            [-0.143540669856454, -7.79904306220095, 0],
                            [28.6124401913875, -3.58851674641148, 0],
                            [25.5502392344497, -7.36842105263157, 0],
                            [8.22966507177034, -12.0095693779904, 0],
                            [1.05263157894737, -11.9617224880383, 0],
                            [-3.15789473684210, -15.7416267942584, 0],
                            [-9.52153110047846, -14.8325358851675, 0],
                            [-22.5837320574162, -7.12918660287081, 0],
                            [-23.1100478468899, 1.29186602870813, 0],
                            [-23.0622009569378, 4.64114832535885, 0],
                            [-23.2057416267942, 7.84688995215311, 0],
                            [-26.9856459330143, -1.91387559808612, 0], 
                            [-0.191387559808609, 6.60287081339713, 0],
                            [2.67942583732058, 5.45454545454545, 0]], dtype=np.float32)
            pts_3d = pts_3d_.copy()
            pts_3d[:, 0] = -pts_3d_[:, 1]
            pts_3d[:, 1] = -pts_3d_[:, 0]

        pts_world = pts_3d[:,:2]

        if sub_id == 0:
            pts_2d = np.array([[1575, 611], [1428, 608], [1256, 605], [1066, 876],
                            [1368, 924], [1866, 601]], dtype=np.float32)
        else:
            ### 20201224
            pts_2d = np.array([[1572.17701863354, 609.071428571429], 
                            [1423.10869565217, 608.077639751553], 
                            [1253.17080745342, 604.102484472050], 
                            [1680.50000000000, 517.642857142857], 
                            [1569.19565217391, 521.618012422360], 
                            [1256.15217391304, 569.319875776398], 
                            [1139.87888198758, 594.164596273292], 
                            [911.307453416149, 606.090062111801], 
                            [755.282608695652, 636.897515527950], 
                            [514.785714285714, 786.959627329193], 
                            [1069.31987577640, 872.425465838509], 
                            [1368.45031055901, 916.152173913044], 
                            [1704.35093167702, 975.779503105590], 
                            [398.512422360248, 975.779503105590], 
                            [1879.25776397516, 614.040372670808],
                            [1832.54968944099, 595.158385093168]], dtype=np.float32) - 1
        ## looks like these points are captured on 1080p screen capture
        pts_2d[:,0] = pts_2d[:,0] / 1920 * img_width
        pts_2d[:,1] = pts_2d[:,1] / 1080 * img_height

        pts_cam = pts_2d
    elif name == "roundabout":
        pts_3d = np.array([[0,0,0], [23.75,0,0],[0,25.75,0],
                [0,-29.25,0],[-18,0,0],[18.75,20,0],[-17,10,0],
                [32.75,-10,0],[-17.25,-15.5,0]], dtype=np.float32)
        pts_world = pts_3d[:,:2]

        pts_2d = np.array([[1068, 593], [741,503],[29,682],
                [1565,549],[1552,730],[293,555],[1140,835],
                [837,477],[1867,655]], dtype=np.float32)
        ## looks like these points are captured on 1080p screen capture
        pts_2d[:,0] = pts_2d[:,0] / 1920 * img_width
        pts_2d[:,1] = pts_2d[:,1] / 1080 * img_height
        pts_cam = pts_2d

    else:
        raise ValueError("cam_name not recognized", name)
    return pts_3d, pts_2d

def load_T(name, cam_id):
    if name == "KoPER":
        assert cam_id in [1,4]
        if cam_id == 1:
            T = np.array([-0.998701024990115, -0.0243198486052637, 0.0447750784199287, 12.0101713926222,
                            -0.0488171908715811, 0.708471062121443, -0.704049455657713, -3.19544493781655,
                            -0.0145994711925239, -0.705320706558607, -0.708738002607851, 18.5953697835002,
                            0, 0, 0, 1], dtype=np.float).reshape(4,4)
            fx=336.2903
            fy=335.5113
            cx=321.3685
            cy=251.1326
        else:
            T = np.array([0.916927873702706, 0.399046485499693, -0.00227526644131446, -9.32023173352383,
                            0.287745260046085, -0.665109208127767, -0.689080841835460, -5.17417993923343,
                            -0.276488588820672, 0.631182733979628, -0.724680906729269, 17.1155540514235,
                            0, 0, 0, 1], dtype=np.float).reshape(4,4)
            fx=331.2292
            fy=330.4413
            cx=325.4500
            cy=252.1456
    else:
        raise ValueError("cam_name not recognized", name)

    return fx, fy, cx, cy, T

def load_spec_dict_bev(u_size, v_size, name, cam_id=None, calib=None):
    assert name in ["lturn", "KoPER", "CARLA", "roundabout", "BrnoCompSpeed", "rounD", "rounD_raw"]
    arg_dict = {}
    arg_dict["u_size"] = u_size
    arg_dict["v_size"] = v_size
    
    if name == "lturn":
        ### copied from trafcam_proc
        if cam_id == 0:
            ### ratio: 17/13, 4x
            x_min = -10
            x_max = 42 #30
            y_min = -37#-30
            y_max = 31

        else:
            # ### 20201224
            # ### ratio: 17/12, 5x
            # x_min = -10
            # x_max = 50 #30
            # y_min = -53#-30
            # y_max = 32

            ### ratio: 21/13, 4x
            x_min = -10
            x_max = 42 #30
            y_min = -53#-30
            y_max = 31

        u_axis="-x"
        v_axis="y"

        arg_dict["x_min"] = x_min
        arg_dict["x_max"] = x_max
        arg_dict["y_min"] = y_min
        arg_dict["y_max"] = y_max
        arg_dict["u_axis"] = u_axis
        arg_dict["v_axis"] = v_axis

    elif name == "KoPER":
        vuratio = float(v_size)/ u_size
        if cam_id == 1:
            x_size = 60
            x_max = 45
            y_min = -30
            u_axis = "-x"
            v_axis = "y"
        elif cam_id == 4:
            x_size = 50
            x_max = 30
            y_min = -13
            u_axis = "x"
            v_axis = "-y"
        else:
            raise ValueError("cam_id not recogized")
        y_size = vuratio * x_size

        arg_dict["x_max"] = x_max
        arg_dict["x_size"] = x_size
        arg_dict["y_min"] = y_min
        arg_dict["y_size"] = y_size
        arg_dict["u_axis"] = u_axis
        arg_dict["v_axis"] = v_axis

    elif name == "roundabout":
        ### copied from trafcam_proc

        if cam_id == 0:
            x_min = -23.97
            y_min = -33.57#-30

            u_axis="-y"
            v_axis="-x"

            arg_dict["x_min"] = x_min
            arg_dict["x_size"] = 60
            arg_dict["y_min"] = y_min
            arg_dict["y_size"] = 60
            arg_dict["u_axis"] = u_axis
            arg_dict["v_axis"] = v_axis
        else:
            x_min = -25#-23.97
            y_min = -43#-33.57#-30

            u_axis="-y"
            v_axis="-x"

            arg_dict["x_min"] = x_min
            arg_dict["x_size"] = 70
            arg_dict["y_min"] = y_min
            arg_dict["y_size"] = 70
            arg_dict["u_axis"] = u_axis
            arg_dict["v_axis"] = v_axis

    elif name == "rounD":

        if cam_id == 0:
            x_min = 0
            y_max = 0
            u_axis = "x"
            v_axis = "-y"
            u_size = 1544
            v_size = 936
            m_per_px = 0.0148098329880904
        elif cam_id == 1:
            u_size = 1678
            v_size = 936
            m_per_px = 0.0136334127882737
        elif cam_id >= 2:
            x_center = 96
            y_center = -17.2
            u_axis = "x"
            v_axis = "-y"
            u_size = 1678
            v_size = 936
            m_per_px = 0.0101601513616589
        else:
            raise ValueError("cam_id {} not recognized. ".format(cam_id))
        
        x_size = u_size * m_per_px * 10
        y_size = v_size * m_per_px * 10     # scale_down_factor in drone-dataset-tools

        if cam_id >= 2:
            x_min = x_center - x_size / 2
            y_max = y_center + y_size / 2

        arg_dict["x_min"] = x_min
        arg_dict["x_size"] = x_size
        arg_dict["y_max"] = y_max
        arg_dict["y_size"] = y_size
        arg_dict["u_axis"] = u_axis
        arg_dict["v_axis"] = v_axis

    elif name == "rounD_raw":
        if cam_id == 2:
            x_min = 0
            y_max = 0
            u_axis = "x"
            v_axis = "-y"
            u_size = 1678
            v_size = 936
            m_per_px = 0.0101601513616589

        x_size = u_size * m_per_px * 10
        y_size = v_size * m_per_px * 10     # scale_down_factor in drone-dataset-tools

        arg_dict["x_min"] = x_min
        arg_dict["x_size"] = x_size
        arg_dict["y_max"] = y_max
        arg_dict["y_size"] = y_size
        arg_dict["u_axis"] = u_axis
        arg_dict["v_axis"] = v_axis

    elif name == "CARLA":
        # if cam_id == 3:
        #     arg_dict['x_min'] = -31.9
        #     arg_dict['y_min'] = -28.8  
        #     arg_dict['x_size'] = 70
        #     arg_dict['y_size'] = 70
        #     # arg_dict['u_size'] = u_size #544
        #     # arg_dict['v_size'] = v_size #544
        #     arg_dict['u_axis'] = "-x"
        #     arg_dict['v_axis'] = "-y"
        # elif cam_id == 2:
        #     arg_dict['x_min'] = -99.25
        #     arg_dict['y_min'] = 110.48
        #     arg_dict['x_size'] = 50
        #     arg_dict['y_size'] = 50
        #     # arg_dict['u_size'] = u_size #544
        #     # arg_dict['v_size'] = v_size #544
        #     arg_dict['u_axis'] = "-x"
        #     arg_dict['v_axis'] = "-y"
        if cam_id in [1]:       ## crossing, low camera, c1, c2
            arg_dict['x_min'] = -101.76161565095929
            arg_dict['y_min'] = 111.53369067537298
            arg_dict['x_size'] = 60
            arg_dict['y_size'] = 60
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "-x"
            arg_dict['v_axis'] = "-y"
        if cam_id in [1.2]:       ## crossing, low camera, c1, c2
            arg_dict['x_min'] = -103.76161565095929
            arg_dict['y_min'] = 113.53369067537298
            arg_dict['x_size'] = 55
            arg_dict['y_size'] = 55
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "-x"
            arg_dict['v_axis'] = "-y"
        elif cam_id in [2.1, 2.9]:
            arg_dict['x_min'] = -104.0157
            arg_dict['y_min'] = -25.07683
            arg_dict['x_size'] = 50
            arg_dict['y_size'] = 50
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "y"
            arg_dict['v_axis'] = "-x"
        elif cam_id in [2.2, 2.8]:
            arg_dict['x_min'] = -99.0157
            arg_dict['y_min'] = -25.07683
            arg_dict['x_size'] = 45
            arg_dict['y_size'] = 45
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "y"
            arg_dict['v_axis'] = "-x"
        elif cam_id in [3.1]:
            arg_dict['x_min'] = 188.47
            arg_dict['y_min'] = -339.23
            arg_dict['x_size'] = 40
            arg_dict['y_size'] = 40
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "y"
            arg_dict['v_axis'] = "-x"
        elif cam_id in [3.2, 3.8]:
            arg_dict['x_min'] = 193.47
            arg_dict['y_min'] = -334.23
            arg_dict['x_size'] = 40
            arg_dict['y_size'] = 40
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "y"
            arg_dict['v_axis'] = "-x"
        elif cam_id in [3.3, 3.9]:
            arg_dict['x_min'] = 193.47
            arg_dict['y_min'] = -339.23
            arg_dict['x_size'] = 45
            arg_dict['y_size'] = 45
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "y"
            arg_dict['v_axis'] = "-x"
        elif cam_id in [4.2, 4.8]:
            arg_dict['x_min'] = -92.72
            arg_dict['y_min'] = -150.01
            arg_dict['x_size'] = 50
            arg_dict['y_size'] = 50
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "-x"
            arg_dict['v_axis'] = "-y"
        elif cam_id in [4.3, 4.4, 4.9]:
            arg_dict['x_min'] = -92.72
            arg_dict['y_min'] = -150.01
            arg_dict['x_size'] = 60
            arg_dict['y_size'] = 60
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "-x"
            arg_dict['v_axis'] = "-y"
        elif cam_id in [5.8, 5.9]:
            arg_dict['x_min'] = -11.43 - 20
            arg_dict['y_min'] = 187.22 - 45
            arg_dict['x_size'] = 70
            arg_dict['y_size'] = 70
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "y"
            arg_dict['v_axis'] = "-x"
        elif cam_id in [6.9]:
            arg_dict['x_min'] = -92.72
            arg_dict['y_min'] = -150.01
            arg_dict['x_size'] = 70
            arg_dict['y_size'] = 70
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "-x"
            arg_dict['v_axis'] = "-y"
        elif cam_id in [7.9]:
            arg_dict['x_min'] = -104.0157
            arg_dict['y_min'] = -45.07683
            arg_dict['x_size'] = 68
            arg_dict['y_size'] = 68
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "x"
            arg_dict['v_axis'] = "y"

    elif name == "BrnoCompSpeed":

        if calib is not None:
            center = calib.gen_center_in_world()
            if cam_id == 0:
                arg_dict['x_min'] = center[0] - 40
                arg_dict['y_min'] = center[1] - 22
                arg_dict['x_size'] = 96
                arg_dict['y_size'] = 48
                arg_dict['u_axis'] = "y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 4.1:
                arg_dict['x_min'] = center[0] - 11
                arg_dict['y_min'] = center[1] - 10
                arg_dict['x_size'] = 52
                arg_dict['y_size'] = 32
                arg_dict['u_axis'] = "y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 4.2:
                arg_dict['x_min'] = center[0] - 12
                arg_dict['y_min'] = center[1] - 10
                arg_dict['x_size'] = 48
                arg_dict['y_size'] = 32
                arg_dict['u_axis'] = "y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 4.3:
                arg_dict['x_min'] = center[0] - 14
                arg_dict['y_min'] = center[1] - 10
                arg_dict['x_size'] = 64
                arg_dict['y_size'] = 28
                arg_dict['u_axis'] = "-y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 5.1:
                arg_dict['x_min'] = center[0] - 13
                arg_dict['y_min'] = center[1] - 10
                arg_dict['x_size'] = 56
                arg_dict['y_size'] = 36
                arg_dict['u_axis'] = "y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 5.2:
                arg_dict['x_min'] = center[0] - 8
                arg_dict['y_min'] = center[1] - 8
                arg_dict['x_size'] = 40
                arg_dict['y_size'] = 24
                arg_dict['u_axis'] = "y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 5.3:
                arg_dict['x_min'] = center[0] - 8
                arg_dict['y_min'] = center[1] - 6
                arg_dict['x_size'] = 56
                arg_dict['y_size'] = 24
                arg_dict['u_axis'] = "-y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 6.1:
                arg_dict['x_min'] = center[0] - 28
                arg_dict['y_min'] = center[1] - 18
                arg_dict['x_size'] = 80
                arg_dict['y_size'] = 40
                arg_dict['u_axis'] = "y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 6.2:
                arg_dict['x_min'] = center[0] - 24
                arg_dict['y_min'] = center[1] - 10
                arg_dict['x_size'] = 80
                arg_dict['y_size'] = 36
                arg_dict['u_axis'] = "y"
                arg_dict['v_axis'] = "-x"   # 4m increment
            elif cam_id == 6.3:
                arg_dict['x_min'] = center[0] - 14
                arg_dict['y_min'] = center[1] - 10
                arg_dict['x_size'] = 64
                arg_dict['y_size'] = 24
                arg_dict['u_axis'] = "-y"
                arg_dict['v_axis'] = "-x"   # 4m increment
        else:
            arg_dict['x_min'] = -34
            arg_dict['y_min'] = -34
            arg_dict['x_size'] = 68
            arg_dict['y_size'] = 68
            # arg_dict['u_size'] = u_size #544
            # arg_dict['v_size'] = v_size #544
            arg_dict['u_axis'] = "x"
            arg_dict['v_axis'] = "y"


    return arg_dict

def R_from_euler_carla(roll, pitch, yaw):
    """This is generate rotation matrix defined in carla coordinated (x,y,z: front,right,up, which is left-handed). 
    The definition of positive rotation angle is the same as counter-clockwise rotations in a right-handed coordinate front-right-down. 
    See https://carla.readthedocs.io/en/latest/python_api/#carla.Rotation """
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180
    
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)

    matrix = np.zeros((3,3))
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def load_calib_from_file_carla(fpath):
    """read from camera calibration file generated by CARLA_scripts repo, each file has one line of: 
    u_size, v_size, fov, x, y, z, roll, pitch, yaw"""

    carla_cam = read_txt_to_array(fpath).reshape(-1)
    cam_intr = carla_cam[:3]
    cam_extr = carla_cam[3:]    # xyzrpy
    u_size = cam_intr[0]
    v_size = cam_intr[1]
    ux = u_size * 0.5
    uy = v_size * 0.5
    fov = cam_intr[2] * np.pi/180
    fx = ux / np.tan(fov*0.5)
    fy = fx
    K = np.array([[fx, 0, ux], [0, fy, uy], [0, 0, 1]], dtype=np.float)

    t = cam_extr[:3]
    roll = cam_extr[3]
    pitch = cam_extr[4]
    yaw = cam_extr[5]

    R = R_from_euler_carla(roll, pitch, yaw)
    T_world_cam = np.concatenate((R, t.reshape(3,1)), axis=1)
    T_world_cam = np.concatenate((T_world_cam, np.array([[0,0,0,1]], dtype=T_world_cam.dtype)), axis=0)
    T_cam_world = np.linalg.inv(T_world_cam)
    coord_permute = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]], dtype=np.float)
    T_cam_world = coord_permute.dot(T_cam_world)

    return K, T_cam_world, u_size, v_size

def load_calib_from_file_blender(fpath):
    data_dict = read_txt_to_dict(fpath)

    Rt = data_dict["cam_pos_inv"].reshape(4,4)
    K = data_dict["K"].reshape(3,4)[:,:3]

    u_size = (K[0,2]*2).round().astype(int)
    v_size = (K[1,2]*2).round().astype(int)
    return K, Rt, u_size, v_size

def load_vps_from_file_BrnoCompSpeed(fpath):
    with open(fpath) as f:
        calibration = json.load(f)

    for key in calibration:
        if isinstance(calibration[key], list):
            calibration[key] = np.array(calibration[key])

    return calibration
