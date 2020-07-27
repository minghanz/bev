import numpy as np

from ..converter.rbox_cvt import blender2world, bev2coco, kitti2world
from ..rbox import rbox_world_bev

# import sys
# sys.path.append("../../")
# from cvo_ops.utils_general.io import read_calib_file

from .utils import read_txt_to_dict

def write_txt_coco(anno_coco, txt_path):
    ### write txt in yolo format 
    ### exclude out-of-view boxes

    with open(txt_path, "w") as f:
        for i in range(anno_coco.shape[0]):
            if anno_coco[i,[1,2]].min() < 0 or anno_coco[i,[1,2]].max() > 1:
                continue
            f.write(" ".join(str(x) for x in anno_coco[i]) + "\n")

def read_txt_yolo_pred(txt_source, rbox_type):
    ### TODO: make sure the yolo prediction format
    assert rbox_type in ["xywhr", "xy8"]

    with open(txt_source) as f:
        lines = f.readlines()
        n_dets = len(lines)

        frame_ids = np.zeros((n_dets))
        dets = np.zeros((n_dets, 5)) if rbox_type == "xywhr" else np.zeros((n_dets, 8))

        for i in range(n_dets):
            det = [float(x) for x in lines[i].split()]
            frame_ids[i] = int(det[0])
            if rbox_type == "xy8":
                dets[i] = np.array(det[2:10])
            else:
                dets[i] = np.array(det[1:6])
    
    return dets, frame_ids

def read_txt_blender(txt_source):
    ##### generate annotation file in the format of yolov3 as described in https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
    txt_content = read_txt_to_dict(txt_source) # read_calib_file
    # with open(txt_target, "w") as f:
    n_obj = len([key for key in txt_content if "center" in key])
    anno_blender = np.zeros((n_obj, 5))
    for i in range(n_obj):
        center = txt_content["center"] if "center" in txt_content else txt_content["center_%d"%i]
        center = center[:2] # x,y in world coordinate
        lwh_yaw_scale = txt_content["lwh_yaw_scale"] if "lwh_yaw_scale" in txt_content else txt_content["lwh_yaw_scale_%d"%i]
        l = lwh_yaw_scale[0]
        w = lwh_yaw_scale[1]
        yaw = lwh_yaw_scale[3]
        anno_blender[i] = center[0],center[1],w,l,yaw

    return anno_blender


def read_txt_blender_to_coco(txt_source, H_world2bev, width, height):
    ##### generate annotation file in the format of yolov3 as described in https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
    anno_blender = read_txt_blender(txt_source)

    anno_world = blender2world(anno_blender)
    anno_bev = rbox_world_bev(anno_world, H_world2bev, src="world")
    anno_coco = bev2coco(anno_bev, width, height)
    
    # txt_content = read_calib_file(txt_source)
    # # with open(txt_target, "w") as f:
    # n_obj = len([key for key in txt_content if "center" in key])
    # anno_coco = np.zeros((n_obj, 6))
    # for i in range(n_obj):
    #     center = txt_content["center"] if "center" in txt_content else txt_content["center_%d"%i]
    #     center = center[:2] # x,y in world coordinate
    #     lwh_yaw_scale = txt_content["lwh_yaw_scale"] if "lwh_yaw_scale" in txt_content else txt_content["lwh_yaw_scale_%d"%i]
    #     l = lwh_yaw_scale[0]
    #     w = lwh_yaw_scale[1]
    #     yaw = lwh_yaw_scale[3]
    #     xywhr = np.array([center[0],center[1],w,l,yaw]).reshape(1,-1)


    #     x,y,w,l,yaw = xywhr_world2bev(center[0],center[1],w,l,yaw,H_world2bev)
    #     anno_coco[i] = 0, x/width, y/height, w/width, l/height, yaw

    return anno_coco

def read_txt_kitti_to_coco(txt_source, H_world2bev, width, height):
    ### kitti cam coordinate: x-right, y-down, z-front
    with open(txt_source) as f:
        lines = f.readlines()

    class_names = [line.split()[0] for line in lines]
    wanted_names = ["Car", "Van", "Truck"]
    wanted_items = [x in wanted_names for x in class_names]

    rbox_hwlxyzr = np.array([[float(x) for x in line.split()[8:15]] for line in lines])
    rbox_hwlxyzr = rbox_hwlxyzr[wanted_items]

    xywhr = kitti2world(rbox_hwlxyzr)
    anno_bev = rbox_world_bev(xywhr, H_world2bev, src="world")
    anno_coco = bev2coco(anno_bev, width, height)

    return anno_coco