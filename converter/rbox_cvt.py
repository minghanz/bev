import numpy as np


def coco2bev(anno_coco, width, height ):
    xywhr = anno_coco[:,1:].copy()
    xywhr[:, [0,2]] *= width
    xywhr[:, [1,3]] *= height

    return xywhr

def bev2coco(xywhr, width, height, classes=None):

    xywhr_new = xywhr.copy()

    xywhr_new[:, [0,2]] /= width
    xywhr_new[:, [1,3]] /= height

    if classes is None:
        ### single class mode, set class_id to zero
        anno_coco = np.concatenate([np.zeros_like(xywhr_new[:,0:1]), xywhr_new], axis=1)
    else:
        anno_coco = np.concatenate([classes.reshape(-1,1), xywhr_new], axis=1)
        
    return anno_coco

def blender2world(xywhr):
    xywhr_new = xywhr.copy()

    xywhr_new[:, 4] += np.pi/2

    return xywhr_new

def world2blender(xywhr):
    xywhr_new = xywhr.copy()

    xywhr_new[:, 4] -= np.pi/2

    return xywhr_new