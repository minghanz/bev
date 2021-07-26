import numpy as np
import cv2
import math


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
    # if H_img_world[2,2] != 0:
    #     H_img_world = H_img_world / H_img_world[2,2]
    # if H_img_world[2].sum() != 0:
    #     H_img_world = H_img_world / H_img_world[2].sum()
    return H_img_world


def homo_from_pts(pts_src, pts_tgt):
    """return a H such that pts_tgt = lambda*H*pts_src
    pts are n*2 np.ndarray
    """
    assert pts_src.ndim==2 and pts_src.shape[1] == 2, pts_src.shape
    assert pts_tgt.ndim==2 and pts_tgt.shape[1] == 2, pts_tgt.shape
    
    H_tgt_src, mask = cv2.findHomography(pts_src, pts_tgt)

    return H_tgt_src
    
def get_focal(vp1, vp2, pp):
    return math.sqrt(- np.dot(vp1[0:2]-pp[0:2], vp2[0:2]-pp[0:2]))  # math.sqrt is faster than np.sqrt? 

def get_K_from_f_pp(focal, pp):
    K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]])
    return K

def get_K_from_vps(vp1, vp2, pp):
    focal = get_focal(vp1, vp2, pp)
    K = get_K_from_f_pp(focal, pp)
    return K, focal

def homo_from_vps(vp1, vp2, height, u_size, v_size, pp=None):
    """vp1, vp2 are arrays of length 2 (u,v)
    Fully automatic roadside camera calibration for traffic surveillance, Dubská, 2015"""
    if pp is None:
        pp = np.array([(u_size-1)*0.5, (v_size-1)*0.5])

    # focal = get_focal(vp1, vp2, pp)
    # K = get_K_from_f_pp(focal, pp)
    K, focal = get_K_from_vps(vp1, vp2, pp)

    vp1W = np.concatenate((vp1, [focal]))    
    vp2W = np.concatenate((vp2, [focal]))    
    ppW = np.concatenate((pp, [0]))
    vp3W = np.cross(vp1W-ppW, vp2W-ppW)

    # vp3Direction = vp3W/vp3W[2]*focal
    # vp3W = vp3Direction + ppW
    # vp3 = vp3W[0:2]

    vp3 = vp3W[0:2]/vp3W[2]*focal + pp
    vp3W = np.concatenate((vp3, [focal]))
    vp3Direction = vp3W-ppW

    vp2Direction = vp2W-ppW
    vp1Direction = vp1W-ppW

    vp3Direction = vp3Direction / np.linalg.norm(vp3Direction)
    vp2Direction = vp2Direction / np.linalg.norm(vp2Direction)
    vp1Direction = vp1Direction / np.linalg.norm(vp1Direction)

    ax_road = np.concatenate((vp3Direction, [-1*height])) # here the scale is the height of camera

    ax1 = np.concatenate((vp1Direction, [0]))
    ax2 = np.concatenate((vp2Direction, [0]))

    M = np.stack((ax1, ax2, ax_road, [0,0,0,1]), axis=0)
    M_inv = np.linalg.inv(M)

    M_inv_43 = M_inv[:, [0, 1, 3]] # 4*3 

    K_34 = np.concatenate((K, np.array([0,0,0]).reshape((3,1))), axis=1)    # 3*4

    H_img_world = np.dot(K_34, M_inv_43)

    return H_img_world

def get_vps_from_homo(H_img_world):
    """pp: principle point"""
    vp1 = np.array([H_img_world[0,0]/H_img_world[2,0], H_img_world[1,0]/H_img_world[2,0]])
    vp2 = np.array([H_img_world[0,1]/H_img_world[2,1], H_img_world[1,1]/H_img_world[2,1]])
    return vp1, vp2

def get_KRt_from_homo(H_img_world, pp):
    """Calculate the intrinsic and extrinsic matrices from homography and camera principle points. """
    vp1, vp2 = get_vps_from_homo(H_img_world)
    K, focal = get_K_from_vps(vp1, vp2, pp)
    R, t = Rt_from_homo_K(H_img_world, K)
    return K, focal, R, t

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
