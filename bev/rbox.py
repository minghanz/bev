import numpy as np
import cv2

from .homo import homo_from_KRt
### bev coordinate
### ----> u
### |
### v
### v

### yaw=0 correspond to vector(0,1)
### positive to counter-clockwise direction, yaw = atan2(sin, cos) = atan2(u, v)

### world coordinate
### nright-handed xy coord, positive yaw counter clockwise from positive x axis

### rbox: x,y,w(width), h(length), r(yaw)
### rbox3d: xyzl(length)w(width)h(height)r(roll)p(pitch)y(yaw)

def v2yaw(x, mode):
    assert mode in ["bev", "world"]
    if mode == "bev":
        y = np.arctan2(x[:,0], x[:,1])
    else:
        y = np.arctan2(x[:,1], x[:,0])

    return y

def yaw2v(x, mode):
    assert mode in ["bev", "world"]
    if mode == "bev":
        y = np.stack((np.sin(x), np.cos(x)), axis=1)
    else:
        y = np.stack((np.cos(x), np.sin(x)), axis=1)

    return y

def yaw2mat(x, mode):
    assert mode in ["bev", "world"]
    x = x.reshape(-1,1)
    if mode == "bev":
        # y = np.stack([np.array([[np.cos(x[i]), np.sin(x[i]) ], [-np.sin(x[i]), np.cos(x[i])]]) for i in range(x.shape[0])], axis=0) # n*2*2
        y = np.concatenate([np.cos(x), np.sin(x), -np.sin(x), np.cos(x)], axis=1).reshape(-1,2,2)
    else:
        # y = np.stack([np.array([[np.cos(x[i]), -np.sin(x[i]) ], [np.sin(x[i]), np.cos(x[i])]]) for i in range(x.shape[0])], axis=0) # n*2*2
        y = np.concatenate([np.cos(x), -np.sin(x), np.sin(x), np.cos(x)], axis=1).reshape(-1,2,2)

    return y

def xy82xywhr(xy8, mode):
    assert mode in ["bev", "world"]

    top_left = xy8[:, 0:2]
    bot_left = xy8[:, 2:4]
    top_right = xy8[:, 6:8]
    w = np.sqrt(((top_right - top_left)**2).sum(1, keepdims=True))
    h = np.sqrt(((bot_left - top_left)**2).sum(1, keepdims=True))
    xy = 0.5*(bot_left + top_right)
    vec = top_left - bot_left
    r = v2yaw(vec, mode).reshape(-1, 1)

    xywhr = np.concatenate([xy, w, h, r], axis=1)
    return xywhr

def xywhr2xyxy(x, mode, external_aa=False):
    assert mode in ["bev", "world"]
    # convert n*5 boxes from [x,y,w,h,yaw] to [x1, y1, x2, y2] if external_aa, 
    # else [x1, y1, x2, y2, x3, y3, x4, y4] from [x_min, y_min] [x_min, y_max] [x_max, y_max], [x_max, y_min]
    if external_aa:
        y = np.zeros((x.shape[0], 4), dtype=x.dtype)
    else:
        y = np.zeros((x.shape[0], 8), dtype=x.dtype)

    # xyxy_ur = xywh2xyxy(x[:,:4])
    ### This is different for two modes because
    ### for "bev", yaw=0 is pos y axis, w corresponds to x variation, h corresponds to y variation
    ### for "world", yaw=0 is pos x axis, w corresponds to y variation, h corresponds to x variation
    if mode == "bev":
        y[:, 0] = - x[:, 2] / 2  # top left x
        y[:, 1] = - x[:, 3] / 2  # top left y
        y[:, 2] = - x[:, 2] / 2  # bottom left x
        y[:, 3] = + x[:, 3] / 2  # bottom left y
        y[:, 4] = + x[:, 2] / 2  # bottom right x
        y[:, 5] = + x[:, 3] / 2  # bottom right y
        y[:, 6] = + x[:, 2] / 2  # top right x
        y[:, 7] = - x[:, 3] / 2  # top right y
    else:
        y[:, 0] = - x[:, 3] / 2
        y[:, 1] = - x[:, 2] / 2
        y[:, 2] = + x[:, 3] / 2
        y[:, 3] = - x[:, 2] / 2
        y[:, 4] = + x[:, 3] / 2
        y[:, 5] = + x[:, 2] / 2 
        y[:, 6] = - x[:, 3] / 2  
        y[:, 7] = + x[:, 2] / 2  
        

    y = y.reshape(-1, 4, 2).swapaxes(1,2) # n*2*4
    rot_mat = yaw2mat(x[:,4], mode)
    y = np.matmul(rot_mat, y)
    y = y.swapaxes(1,2).reshape(-1, 8)

    y += x[:, [0,1,0,1,0,1,0,1]]    # n*8
    if not external_aa:
        return y
    else:
        x_min = y[:,[0,2,4,6]].min(axis=1) # n*1
        x_max = y[:,[0,2,4,6]].max(axis=1) # n*1
        y_min = y[:,[1,3,5,7]].min(axis=1) # n*1
        y_max = y[:,[1,3,5,7]].max(axis=1) # n*1
        y = np.stack([x_min, y_min, x_max, y_max], axis=1)
        return y

def xywhr2xyvec(xywhr, mode):
    """return a vector [x_start, y_start, x_end, y_end]"""
    assert mode in ["bev", "world"]

    vs = yaw2v(xywhr[:, 4], mode)           # normalized directional vector
    lvs = xywhr[:, 3:4]                     # length
    vs = vs * lvs                           # scale the vector
    xs = xywhr[:, 0]
    ys = xywhr[:, 1]

    xyvec = np.stack([xs, ys, xs+vs[:,0], ys+vs[:,1]], axis=1)
    return xyvec

def xy82xyvec(xy8):
    """return a vector [x_start, y_start, x_end, y_end]"""
    vs = xy8[:, 2:4] - xy8[:, :2]
    xs = 0.5*(xy8[:, 0] + xy8[:, 4])
    ys = 0.5*(xy8[:, 1] + xy8[:, 5])

    xyvec = np.stack([xs, ys, xs+vs[:,0], ys+vs[:,1]], axis=1)
    return xyvec

def pts_world_bev(pts_src, H):
    pts_src = np.array(pts_src)
    if pts_src.ndim == 1:
        pts_src = pts_src[None, :]

    homo = True
    if pts_src.shape[1] == 2:
        homo = False
        pts_src = np.concatenate([pts_src, np.ones_like(pts_src[:, [0]])], axis=1)
    assert pts_src.shape[1] == 3

    pts_tgt = H.dot(pts_src.T).T    # N*3
    pts_tgt = pts_tgt / pts_tgt[:,[2]]
    if not homo:
        pts_tgt = pts_tgt[:, :2]
    return pts_tgt

def dist_world_bev(dist_src, H):
    scale = np.sqrt(H[0,0]**2 + H[1,0]**2)
    scale_1 = np.sqrt(H[0,1]**2 + H[1,1]**2)
    assert np.abs(scale - scale_1) < 1e-5

    dist_tgt = scale * dist_src

    return dist_tgt

def angle_world_bev(angle_src, H, src):
    assert src in ["bev", "world"]
    target = "world" if src == "bev" else "bev"

    angle = np.array(angle_src).reshape(-1)     # 1 or more elements
    v_src = np.concatenate([yaw2v(angle, src), np.zeros_like(angle)[...,None]], axis=1) # n*3 homogeneous coord, normal yaw definition (right-handed xy coord, positive yaw counter clockwise from positive x axis)
    v_tgt = H.dot(v_src.T).T[:,:2] # n*3
    r_tgt = v2yaw(v_tgt, target)

    return r_tgt

def rbox_world_bev(rbox_src, H, src):
    """src is "bev" or "world". 
    rbox_src is n*5 array
    H represent a similarity transform (translation, rotation, reflection, scaling) https://en.wikipedia.org/wiki/Similarity_(geometry) 
    Therefore of form 
    [ s*cos(a), -s*sin(a), cx
      s*sin(a), s*cos(a),  cy
      0,        0,         1  ] or 
    [ s*cos(a), s*sin(a), cx
      s*sin(a), -s*cos(a),  cy
      0,        0,         1  ] (with reflection) 
    The definition of world coordinate can be arbitrary, as long as the H_world_bev (H_bev_world) embedded the correspondence between the world coordinate and bev. 
    The only thing that needs attention is the definition of yaw in world should be (right-handed xy coord, positive yaw counter clockwise from positive x axis), otherwise the calculated yaw in bev will be wrong.
      """

    assert src in ["bev", "world"]
    target = "world" if src == "bev" else "bev"

    #### normalize H so that we do not need to normalize result after matrix multiplication
    H = H / H[2,2]
    assert np.abs(H[2,0]) + np.abs(H[2,1]) < 1e-5

    if len(rbox_src) == 0:
        # print("rbox_world_bev: Empty label", rbox_src.shape)
        return rbox_src
    #### rotation
    r_src = rbox_src[:, 4]

    # ### result should be the same as [cos(r_src), sin(r_src)]
    # rot_mat = np.array([np.array([np.cos(ri), -np.sin(ri), np.sin(ri), np.cos(ri)]).reshape(2,2) for ri in r_src]) # n*2*2
    # v_local = np.stack([np.ones_like(r_src), np.zeros_like(r_src)], axis=1)[...,None] # n*2*1
    # v_src = rot_mat.dot(v_local)

    r_tgt = angle_world_bev(r_src, H, src)

    #### xy
    xy_tgt = pts_world_bev(rbox_src[:, :2], H)
    # xy_src = np.concatenate([rbox_src[:, :2], np.ones_like(r_src)[...,None]], axis=1) # n*3
    # xy_tgt = H.dot(xy_src.T).T[:, :2] # n*3

    #### wh
    wh_src = rbox_src[:, 2:4]
    wh_tgt = dist_world_bev(wh_src, H)

    rbox_tgt = np.concatenate([xy_tgt, wh_tgt, r_tgt[...,None]], axis=1)

    return rbox_tgt

def rbox_world_img(rbox_world, H_img_world):
    xy_world = rbox_world[:, :2]

    xy_img = pts_world_bev(xy_world, H_img_world)

    return xy_img

def rbox_zt2tt_world(rboxzt, K, Rt):
    """This function converts xywhrt to xywhrtt in world space"""
    H_cam_world = homo_from_KRt(K=K, Rt_homo=Rt)
    H_world_cam = np.linalg.inv(H_cam_world)

    xyz_low = rboxzt[:,[0,1,5]].T     # 3*N
    xyz_low_cam = Rt[:3,:3].dot(xyz_low) + Rt[:3,[3]]
    uvd_low = K.dot(xyz_low_cam)
    uv1_low = uvd_low / np.clip(uvd_low[2], a_min=1e-2, a_max=None)

    xy1_low = H_world_cam.dot(uv1_low)    # 3*N
    xy1_low = xy1_low / xy1_low[2]
    assert (xy1_low[2]==1).all(), "{}".format(xy1_low)

    xyz_high = rboxzt[:,[0,1,5]].T     # 3*N
    xyz_high[2] = xyz_high[2] + rboxzt[:,6].T
    xyz_high_cam = Rt[:3,:3].dot(xyz_high) + Rt[:3,[3]]
    uvd_high = K.dot(xyz_high_cam)
    uv1_high = uvd_high / np.clip(uvd_high[2], a_min=1e-2, a_max=None)

    xy1_high = H_world_cam.dot(uv1_high)    # 3*N
    xy1_high = xy1_high / xy1_high[2]
    assert (xy1_high[2]==1).all(), "{}".format(xy1_high)

    dudv = xy1_high[:2] - xy1_low[:2]
    
    rboxtt = np.concatenate([xy1_low[:2].T, rboxzt[:,2:5], dudv.T], axis=1)

    return rboxtt

def rboxtt_world_bev(rbox_src, H, src):
    assert src in ["bev", "world"]
    target = "world" if src == "bev" else "bev"

    if len(rbox_src) == 0:
        # print("rboxtt_world_bev: Empty label", rbox_src.shape)
        return rbox_src
    
    #### normalize H so that we do not need to normalize result after matrix multiplication
    H = H / H[2,2]
    assert np.abs(H[2,0]) + np.abs(H[2,1]) < 1e-5

    assert rbox_src.shape[1] == 7

    xy1_src = np.concatenate([rbox_src[:, :2], np.ones((rbox_src.shape[0], 1))], axis=1).T
    xy1_tgt = H.dot(xy1_src)      # 3*N
    assert (xy1_tgt[2]==1).all(), "{}".format(xy1_tgt)      # no need to normalize the coordinate because the H here only has scaling and 90degree-wise rotations

    xy1_end_src = np.concatenate([rbox_src[:, :2] + rbox_src[:,5:], np.ones((rbox_src.shape[0], 1))], axis=1).T
    xy1_end_tgt = H.dot(xy1_end_src)      # 3*N
    assert (xy1_end_tgt[2]==1).all(), "{}".format(xy1_end_tgt)

    dudv = xy1_end_tgt - xy1_tgt
    dudv = dudv[:2].T               # N*2

    rbox_xywhr_src = rbox_src[:, :5]
    rbox_xywhr_tgt = rbox_world_bev(rbox_xywhr_src, H, src)

    rboxtt_tgt = np.concatenate([rbox_xywhr_tgt, dudv], axis=1)     # N*7

    return rboxtt_tgt


def rboxzt_world_bev(rbox_src, H, K, Rt, src):
    """This function converts between xywhrt in world space and xywhrXY in bev space"""

    assert src in ["bev", "world"]
    target = "world" if src == "bev" else "bev"

    if len(rbox_src) == 0:
        # print("rboxzt_world_bev: Empty label", rbox_src.shape)
        return rbox_src

    #### normalize H so that we do not need to normalize result after matrix multiplication
    H = H / H[2,2]
    assert np.abs(H[2,0]) + np.abs(H[2,1]) < 1e-5
    
    assert rbox_src.shape[1] == 7

    if src == "world":
        rboxtt = rbox_zt2tt_world(rbox_src, K, Rt)
        
        rboxtt_bev = rboxtt_world_bev(rboxtt, H, src)

        return rboxtt_bev
    else:
        raise NotImplementedError("rboxzt_world_bev only supports converting from world to bev")
        # dudv = rbox_src[:, 5:7].T     # 2*N
        # xy_bev = rbox_src[:, :2].T    # 2*N
        # xy1_bev = np.concatenate([xy_bev, np.ones((1, xy_bev.shape[1]))], axis=0)
        # xy1_world = 

        # uv1_bev = 