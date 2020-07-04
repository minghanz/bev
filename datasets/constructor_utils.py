import numpy as np

def load_pts(name, img_width, img_height):
    if name == "lturn":
        ### load homographic calibration and draw on picture
        pts_3d = np.array( [[0, 0, 0], [3.7, 0, 0], [7.4, 0, 0], [-0.87, 21.73, 0],
                        [-4.26, 21.73, 0], [-5.22, -2.17, 0]], dtype=np.float32)
        pts_world = pts_3d[:,:2]

        pts_2d = np.array([[1575, 611], [1428, 608], [1256, 605], [1066, 876],
                        [1368, 924], [1866, 601]], dtype=np.float32)
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

def load_spec_dict_bev(u_size, v_size, name, cam_id=None):
    assert name in ["lturn", "KoPER"]
    arg_dict = {}
    arg_dict["u_size"] = u_size
    arg_dict["v_size"] = v_size
    
    if name == "lturn":
        ### copied from trafcam_proc
        x_min = -10
        x_max = 42 #30
        y_min = -37#-30
        y_max = 31

        u_axis="-x"
        v_axis="y"

        arg_dict["x_min"] = x_min
        arg_dict["x_max"] = x_max
        arg_dict["y_min"] = y_min
        arg_dict["y_max"] = y_max
        arg_dict["u_axis"] = u_axis
        arg_dict["v_axis"] = v_axis

    else:
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

    return arg_dict