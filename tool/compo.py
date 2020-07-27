import numpy as np
import cv2
from ..homo import homo_from_KRt

def composite_reg_img(bg, fg, fg_mask, bw_mode=False):
    if isinstance(bg, str):
        bg = cv2.imread(bg)
    if isinstance(fg, str):
        fg = cv2.imread(fg)
    if isinstance(fg_mask, str):
        fg_mask = cv2.imread(fg_mask)

    if bw_mode:
        fg = cv2.cvtColor(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)#[:,:,np.newaxis]

    bg = bg.astype(np.float)
    fg = fg.astype(np.float)
    fg_mask = fg_mask.astype(np.float)/255
    
    compo = fg * fg_mask + bg * (1-fg_mask)
    compo = compo.round()
    compo[compo>255] = 255
    compo = compo.astype(np.uint8)
    return compo

def composite_bev_img(bg, fg, fg_mask, H_world2bev, H_img2world_fix, K, RT, x_size, y_size, bw_mode=False):
    if isinstance(bg, str):
        bg = cv2.imread(bg)
    if isinstance(fg, str):
        fg = cv2.imread(fg)
    if isinstance(fg_mask, str):
        fg_mask = cv2.imread(fg_mask)

    if bw_mode:
        fg = cv2.cvtColor(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)#[:,:,np.newaxis]

    H_img2bev_fix = H_world2bev.dot(H_img2world_fix)
    bg_bev = cv2.warpPerspective(bg, H_img2bev_fix, (x_size, y_size))

    # H_world2img_cam = K.dot(RT)[:, [0,1,3]]
    H_world2img_cam = homo_from_KRt(K, Rt_homo=RT)

    H_img2world_cam = np.linalg.inv(H_world2img_cam)
    H_img2bev_cam = H_world2bev.dot(H_img2world_cam)

    fg_bev = cv2.warpPerspective(fg, H_img2bev_cam, (x_size, y_size))
    fg_mask_bev = cv2.warpPerspective(fg_mask, H_img2bev_cam, (x_size, y_size))

    compo = composite_reg_img(bg_bev, fg_bev, fg_mask_bev)
    return compo, H_world2img_cam