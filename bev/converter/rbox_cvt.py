import numpy as np

from ..rbox import yaw2v

def coco2bev(anno_coco, width, height ):
    if len(anno_coco) == 0:
        print("coco2bev: Empty label", anno_coco.shape)
        return anno_coco

    xywhr = anno_coco[:,1:].copy()
    xywhr[:, [0,2]] *= width
    xywhr[:, [1,3]] *= height

    return xywhr

def bev2coco(xywhr, width, height, classes=None):
    if len(xywhr) == 0:
        print("bev2coco: Empty label", xywhr.shape)
        return xywhr

    xywhr_new = xywhr.copy()

    xywhr_new[:, [0,2]] /= width
    xywhr_new[:, [1,3]] /= height

    if classes is None:
        ### single class mode, set class_id to zero
        anno_coco = np.concatenate([np.zeros_like(xywhr_new[:,0:1]), xywhr_new], axis=1)
    else:
        anno_coco = np.concatenate([classes.reshape(-1,1), xywhr_new], axis=1)
        
    return anno_coco

def coco2bev_rboxtt(anno_coco, width, height ):
    if len(anno_coco) == 0:
        print("coco2bev_rboxtt: Empty label", anno_coco.shape)
        return anno_coco

    xywhr = anno_coco[:,1:].copy()
    xywhr[:, [0,2,5]] *= width
    xywhr[:, [1,3,6]] *= height

    return xywhr

def bev2coco_rboxtt(xywhrt, width, height, classes=None):
    if len(xywhrt) == 0:
        print("bev2coco_rboxtt: Empty label", xywhrt.shape)
        return xywhrt

    xywhr_new = xywhrt.copy()

    xywhr_new[:, [0,2,5]] /= width
    xywhr_new[:, [1,3,6]] /= height

    if classes is None:
        ### single class mode, set class_id to zero
        anno_coco = np.concatenate([np.zeros_like(xywhr_new[:,0:1]), xywhr_new], axis=1)
    else:
        anno_coco = np.concatenate([classes.reshape(-1,1), xywhr_new], axis=1)
        
    return anno_coco

def blender2world(xywhr):
    if len(xywhr) == 0:
        print("blender2world: Empty label", xywhr.shape)
        return xywhr
    xywhr_new = xywhr.copy()

    xywhr_new[:, 4] += np.pi/2

    return xywhr_new

def blender2world_rboxzt(xywhr):
    return blender2world(xywhr)

def world2blender(xywhr):
    if len(xywhr) == 0:
        print("world2blender: Empty label", xywhr.shape)
        return xywhr
    xywhr_new = xywhr.copy()

    xywhr_new[:, 4] -= np.pi/2

    return xywhr_new

def kitti2world(rbox_hwlxyzr):
    if len(rbox_hwlxyzr) == 0:
        print("kitti2world: Empty label", rbox_hwlxyzr.shape)
        return rbox_hwlxyzr
    ### kitti cam coordinate: x-right, y-down, z-front, r=0 when vehicle facing right, facing front as 90 degree 

    xywhr = rbox_hwlxyzr[:, [3,5,1,2,6]].copy()
    xywhr[:,-1] = -xywhr[:,-1]

    return xywhr

def KoPER2world(rbox_xywhr):
    if len(rbox_xywhr) == 0:
        print("KoPER2world: Empty label", rbox_xywhr.shape)
        return rbox_xywhr
    ##### adjust rbox position, because KoPER does not annotate the center, but the front point
    h = rbox_xywhr[:,3]
    v = yaw2v(rbox_xywhr[:,4], "world")
    rbox_new = rbox_xywhr.copy()
    rbox_new[:, 0] -= v[:,0] * h /2 
    rbox_new[:, 1] -= v[:,1] * h /2 

    return rbox_new

def carla2world(rbox_9d):
    if len(rbox_9d) == 0:
        print("carla2world: Empty label", rbox_9d.shape)
        return rbox_9d
    ##### xyzlwhrpy
    xywhr = rbox_9d[:, [0,1,4,3,8]]
    # xywhr[:, -1] = np.pi/2 +xywhr[:, -1]

    valid_mask = (np.abs(xywhr[:, 0]) > 1e-3) & (np.abs(xywhr[:, 1]) > 1e-3)
    xywhr = xywhr[valid_mask]   ### this is because invalid vehicles in CARLA will have location (0,0,*)
    return xywhr

def carla2world_rboxzt(rbox_9d):
    """From: xyzlwhrpy
    To: x,y,width,length,yaw,z,height"""
    if len(rbox_9d) == 0:
        print("carla2world: Empty label", rbox_9d.shape)
        return rbox_9d
    ##### xyzlwhrpy
    xywhrzt = rbox_9d[:, [0,1,4,3,8,2,5]]   # to x,y,width,length,yaw,height

    # xywhrzt[:,6] = rbox_9d[:,2] + rbox_9d[:,5]/2
    # xywhrzt[:,5] = 0

    xywhrzt[:,5] = rbox_9d[:,2] - rbox_9d[:,5]/2

    # print(rbox_9d[:,[2,5]])

    valid_mask = (np.abs(xywhrzt[:, 0]) > 1e-3) & (np.abs(xywhrzt[:, 1]) > 1e-3)
    xywhrzt = xywhrzt[valid_mask]   ### this is because invalid vehicles in CARLA will have location (0,0,*)

    return xywhrzt