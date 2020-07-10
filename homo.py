import numpy as np
import cv2


def homo_from_KRt(K, R=None, t=None, Rt_homo=None):
    """we assume that in the world coordinate z=0 represent the plane
    the Rt is such that pt_target = H*pt_source, H = K(3*3)*Rt[:3, [0,1,3]]"""
    if K.shape[1] == 4:
        K = K[:, :3]
    if Rt_homo is not None:
        assert R is None and t is None
        Rt = Rt_homo[:, [0,1,3]]
        Rt = Rt[:3]
    else:
        assert R is not None and t is not None
        if t.ndim == 1:
            t = t.reshape(-1,1)
        Rt = np.concatenate((R[:,[0,1]],t),axis=1)

    H_img_world = K.dot(Rt)
    # if H[2,2] != 0:
    #     H = H / H[2,2]
    return H_img_world


def homo_from_pts(pts_src, pts_tgt):
    """return a H such that pts_tgt = lambda*H*pts_src
    pts are n*2 np.ndarray
    """
    assert pts_src.ndim==2 and pts_src.shape[1] == 2, pts_src.shape
    assert pts_tgt.ndim==2 and pts_tgt.shape[1] == 2, pts_tgt.shape
    
    H_tgt_src, mask = cv2.findHomography(pts_src, pts_tgt)

    return H_tgt_src
    

def Rt_from_homo_K(H_img_world, K):
    """https://docs.opencv.org/master/d9/dab/tutorial_homography.html"""
    H_img_world = np.linalg.inv(K).dot(H_img_world)

    norm_H1 = np.sqrt((H_img_world[:,0]**2).sum())
    H_img_world = H_img_world / norm_H1


    R1 = H_img_world[:,0]
    R2 = H_img_world[:,1]
    tvec = H_img_world[:,2]
    R3 = np.cross(R1, R2)

    R = np.stack((R1, R2, R3), axis=1)
    w, u, vt = cv2.SVDecomp(R)
    R = np.matmul(u, vt)

    return R, tvec

def Rt_from_pts_K_dist(pts_world, pts_img, K, dist_coeffs):
    """https://docs.opencv.org/master/d9/dab/tutorial_homography.html"""
    retval, rvec, tvec = cv2.solvePnP(pts_world, pts_img, K, dist_coeffs)

    R, jacobian	= cv2.Rodrigues(rvec)
    return R, tvec
