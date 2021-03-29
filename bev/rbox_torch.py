################## difference to numpy version:
######## concatenate -> cat
######## axis -> dim
######## add device=
######## dot -> mm
######## swapaxes -> transpose
######## transpose -> permute

import torch
import cv2

### bev coordinate
### ----> u
### |
### v
### v

### yaw=0 correspond to vector(0,1)
### positive to counter-clockwise direction, yaw = atan2(sin, cos) = atan2(u, v)

### world coordinate
### nright-handed xy coord, positive yaw counter clockwise from positive x dim

def v2yaw(x, mode):
    assert mode in ["bev", "world"]
    if mode == "bev":
        y = torch.arctan2(x[:,0], x[:,1])
    else:
        y = torch.arctan2(x[:,1], x[:,0])

    return y

def yaw2v(x, mode):
    assert mode in ["bev", "world"]
    if mode == "bev":
        y = torch.stack((torch.sin(x), torch.cos(x)), dim=1)
    else:
        y = torch.stack((torch.cos(x), torch.sin(x)), dim=1)

    return y

def yaw2mat(x, mode):
    assert mode in ["bev", "world"]
    x = x.reshape(-1,1)
    if mode == "bev":
        y = torch.cat([torch.cos(x), torch.sin(x), -torch.sin(x), torch.cos(x)], dim=1).reshape(-1,2,2)
    else:
        y = torch.cat([torch.cos(x), -torch.sin(x), torch.sin(x), torch.cos(x)], dim=1).reshape(-1,2,2)

    return y

def xywhr2xyxy(x, mode, external_aa=False):
    assert mode in ["bev", "world"]
    # convert n*5 boxes from [x,y,w,h,yaw] to [x1, y1, x2, y2] if external_aa, 
    # else [x1, y1, x2, y2, x3, y3, x4, y4] from [x_min, y_min] [x_min, y_max] [x_max, y_max], [x_max, y_min]
    if external_aa:
        y = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
    else:
        y = torch.zeros((x.shape[0], 8), dtype=x.dtype, device=x.device)

    # xyxy_ur = xywh2xyxy(x[:,:4])
    ### This is different for two modes because
    ### for "bev", yaw=0 is pos y dim, w corresponds to x variation, h corresponds to y variation
    ### for "world", yaw=0 is pos x dim, w corresponds to y variation, h corresponds to x variation
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
        

    y = y.reshape(-1, 4, 2).transpose(1,2) # n*2*4
    rot_mat = yaw2mat(x[:,4], mode)
    y = torch.matmul(rot_mat, y)
    y = y.transpose(1,2).reshape(-1, 8)

    y += x[:, [0,1,0,1,0,1,0,1]]    # n*8
    if not external_aa:
        return y
    else:
        x_min = y[:,[0,2,4,6]].min(dim=1) # n*1
        x_max = y[:,[0,2,4,6]].max(dim=1) # n*1
        y_min = y[:,[1,3,5,7]].min(dim=1) # n*1
        y_max = y[:,[1,3,5,7]].max(dim=1) # n*1
        y = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        return y

def xywhr2xyvec(xywhr, mode):
    """return a vector [x_start, y_start, x_end, y_end]"""
    assert mode in ["bev", "world"]

    vs = yaw2v(xywhr[:, 4], mode)           # normalized directional vector
    lvs = xywhr[:, 3:4]                     # length
    vs = vs * lvs                           # scale the vector
    xs = xywhr[:, 0]
    ys = xywhr[:, 1]

    xyvec = torch.stack([xs, ys, xs+vs[:,0], ys+vs[:,1]], dim=1)
    return xyvec

def xy82xyvec(xy8):
    """return a vector [x_start, y_start, x_end, y_end]"""
    vs = xy8[:, 2:4] - xy8[:, :2]
    xs = 0.5*(xy8[:, 0] + xy8[:, 4])
    ys = 0.5*(xy8[:, 1] + xy8[:, 5])

    xyvec = torch.stack([xs, ys, xs+vs[:,0], ys+vs[:,1]], dim=1)
    return xyvec

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
      0,        0,         1  ] (with reflection) """

    assert src in ["bev", "world"]
    target = "world" if src == "bev" else "bev"

    #### normalize H so that we do not need to normalize result after matrix multiplication
    H = H / H[2,2]
    assert torch.abs(H[2,0]) + torch.abs(H[2,1]) < 1e-5

    #### rotation
    r_src = rbox_src[:, 4]

    # ### result should be the same as [cos(r_src), sin(r_src)]
    # rot_mat = torch.array([torch.array([torch.cos(ri), -torch.sin(ri), torch.sin(ri), torch.cos(ri)]).reshape(2,2) for ri in r_src]) # n*2*2
    # v_local = torch.stack([torch.ones_like(r_src), torch.zeros_like(r_src)], dim=1)[...,None] # n*2*1
    # v_src = rot_mat.mm(v_local)

    v_src = torch.cat([yaw2v(r_src, src), torch.zeros_like(r_src)[...,None]], dim=1) # n*3 homogeneous coord, normal yaw definition (right-handed xy coord, positive yaw counter clockwise from positive x dim)
    v_tgt = H.mm(v_src.T).T[:,:2] # n*3
    r_tgt = v2yaw(v_tgt, target)

    #### xy
    xy_src = torch.cat([rbox_src[:, :2], torch.ones_like(r_src)[...,None]], dim=1) # n*3
    xy_tgt = H.mm(xy_src.T).T[:, :2] # n*3

    #### wh
    scale = torch.sqrt(H[0,0]**2 + H[0,1]**2)
    scale_1 = torch.sqrt(H[1,0]**2 + H[1,1]**2)
    assert torch.abs(scale - scale_1) < 1e-5

    wh_src = rbox_src[:, 2:4]
    wh_tgt = wh_src * scale

    rbox_tgt = torch.cat([xy_tgt, wh_tgt, r_tgt[...,None]], dim=1)

    return rbox_tgt