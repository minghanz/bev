import numpy as np
import cv2

def grid_pts_pair_from_grid_coords(grid_x, grid_y):
    x_min = grid_x.min()
    x_max = grid_x.max()
    y_min = grid_y.min()
    y_max = grid_y.max()
    grid_pts_pairs = []
    for x in grid_x:
        pts_pair = [[x, y_min], [x, y_max]]
        grid_pts_pairs.append(pts_pair)
    for y in grid_y:
        pts_pair = [[x_min, y], [x_max, y]]
        grid_pts_pairs.append(pts_pair)
    grid_pts_pairs = np.array(grid_pts_pairs)
    return grid_pts_pairs

def draw_homography_tool(img, H_img_world, grid_x, grid_y):
    grid_color = (0,0,0)
    grid_pts_pairs = grid_pts_pair_from_grid_coords(grid_x, grid_y) # n*2*2
    grid_pts_pairs = np.dstack((grid_pts_pairs, np.ones((grid_pts_pairs.shape[0], 2, 1), dtype=grid_pts_pairs.dtype)) ) # n*2*3
    grid_pts_pairs = grid_pts_pairs.reshape(-1, 3).transpose() # 4*n

    grid_pts_pairs_proj = H_img_world.dot(grid_pts_pairs)
    grid_pts_pairs_proj = grid_pts_pairs_proj / grid_pts_pairs_proj[2]  # 3*n
    grid_pts_pairs_proj = grid_pts_pairs_proj[:2].transpose().reshape(-1, 2, 2) # n*2*2
    grid_pts_pairs_proj = np.round(grid_pts_pairs_proj).astype(int)
    for i in range(grid_pts_pairs_proj.shape[0]):
        cv2.line(img, tuple(grid_pts_pairs_proj[i,0]), tuple(grid_pts_pairs_proj[i,1]), grid_color)

    axis_pts_pairs = np.array([[0,0], [1,0], [0,0], [0,1]], dtype=np.float32)

    axis_pts_pairs_proj = cv2.perspectiveTransform(axis_pts_pairs[np.newaxis], H_img_world)[0].round().astype(int)
    axis_pts_pairs_proj = axis_pts_pairs_proj.reshape(2,2,2)
    cv2.arrowedLine(img, tuple(axis_pts_pairs_proj[0,0]), tuple(axis_pts_pairs_proj[0,1]), (0,0,200))   # red
    cv2.arrowedLine(img, tuple(axis_pts_pairs_proj[1,0]), tuple(axis_pts_pairs_proj[1,1]), (0,0,200))   # red

    cv2.putText(img, "x", tuple(axis_pts_pairs_proj[0,1]+2), 0, 0.4, (0,0,255))
    cv2.putText(img, "y", tuple(axis_pts_pairs_proj[1,1]+2), 0, 0.4, (0,0,255))

    return img

def draw_bev_axis(img, pt_lt_corner, pt_lb_corner, pt_rt_corner):

    u_vec = pt_rt_corner - pt_lt_corner
    v_vec = pt_lb_corner - pt_lt_corner

    u_norm = np.sqrt((u_vec**2).sum())
    v_norm = np.sqrt((v_vec**2).sum())
    
    u_vec = u_vec / u_norm * 15
    v_vec = v_vec / v_norm * 15

    ori = pt_lt_corner.round().astype(int)
    u_end = (pt_lt_corner + u_vec).round().astype(int)
    v_end = (pt_lt_corner + v_vec).round().astype(int)
    

    cv2.arrowedLine(img, tuple(ori), tuple(u_end), (0,200,0))   # green
    cv2.arrowedLine(img, tuple(ori), tuple(v_end), (0,200,0))   # green

    cv2.putText(img, "u", tuple(u_end+2), 0, 0.4, (0,255,0))
    cv2.putText(img, "v", tuple(v_end+2), 0, 0.4, (0,255,0))

    return img

def vis_bspec_and_calib_in_grid(img, bspec, calib=None):
    """if calib is None, draw on bev image"""

    grid_x = np.linspace(bspec.x_min, bspec.x_max, 6)
    grid_y = np.linspace(bspec.y_min, bspec.y_max, 6)

    if calib is not None:
        H_world_img = calib.gen_H_world_img(calib.mode)
    else:
        H_world_img = bspec.gen_H_world_bev()

    H_img_world = np.linalg.inv(H_world_img)
    img = draw_homography_tool(img, H_img_world, grid_x, grid_y)

    if calib is not None:
        pts_world = bspec.gen_bev_corners_in_world()
        pts_img = cv2.perspectiveTransform(pts_world[None], H_img_world)[0]
        pt_lt_corner = pts_img[0]
        pt_lb_corner = pts_img[1]
        pt_rt_corner = pts_img[3]

        img = draw_bev_axis(img, pt_lt_corner, pt_lb_corner, pt_rt_corner)

    return img