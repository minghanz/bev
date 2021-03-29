import bev
import cv2
import numpy as np

from bev.constructor.homo_constr import load_calib, preset_bspec
from bev.visualizer.homo_vis import vis_bspec_and_calib_in_grid
if __name__ == "__main__":

    # img_path = "/media/sda1/datasets/extracted/KoPER/added/SK_1_empty_road_bev.png"
    # # img_path = "/media/sda1/datasets/extracted/KoPER/Sequence1a/KAB_SK_1_undist/KAB_SK_1_undist_1384779301359985.bmp"
    # # img_path = "/media/sda1/datasets/extracted/KoPER/Sequence1a/KAB_SK_4_undist/KAB_SK_4_undist_1384779301359985.bmp"
    # img = cv2.imread(img_path)

    # img = vis_bspec_and_calib_in_grid(img, bspec)
    # # img = vis_bspec_and_calib_in_grid(img, bspec, calib)
    # cv2.imshow("img", img)
    # cv2.waitKey()

    video_tag = "6_left"
    cam_id = int(video_tag[0])
    cam_sub = video_tag.split("_")[1]
    sub_id_dict = {"left":0.1, "center":0.2, "right":0.3}
    cam_id = cam_id + sub_id_dict[cam_sub]
    # print("cam_id", cam_id)

    video_path = "/media/sda1/datasets/extracted/BrnoCompSpeed/dataset/session{}/video.avi".format(video_tag)
    calib_path = "/media/sda1/datasets/extracted/BrnoCompSpeed/calib/session{}/system_dubska_optimal_calib.json".format(video_tag)

    video = cv2.VideoCapture(video_path)    
    calib = load_calib("BrnoCompSpeed", calib_path)
    bspec = preset_bspec("BrnoCompSpeed", cam_id, calib)

    H_world_img = calib.gen_H_world_img()
    H_world_bev =  bspec.gen_H_world_bev()
    H_bev_img = np.linalg.inv(H_world_bev).dot(H_world_img)

    center_world = calib.gen_center_in_world()
    print("center_world", center_world)

    corners = bspec.gen_bev_corners_in_world()
    print("corners", corners)

    new_u = 852
    new_v = 480
    calib_small = calib.scale(align_corners=False, new_u=new_u, new_v=new_v)
    center_world_small = calib_small.gen_center_in_world()
    print("center_world_small", center_world_small)

    H_world_img_small = calib_small.gen_H_world_img()
    H_bev_img_small = np.linalg.inv(H_world_bev).dot(H_world_img_small)

    while video.isOpened():
        _, img = video.read()
        bev = cv2.warpPerspective(img, H_bev_img, (bspec.u_size, bspec.v_size))

        img = vis_bspec_and_calib_in_grid(img, bspec, calib)
        bev = vis_bspec_and_calib_in_grid(bev, bspec)

        cv2.imshow("img", img)
        cv2.imshow("bev", bev)

        img_small = cv2.resize(img, (new_u, new_v))
        bev_small = cv2.warpPerspective(img_small, H_bev_img_small, (bspec.u_size, bspec.v_size))

        img_small = vis_bspec_and_calib_in_grid(img_small, bspec, calib)
        bev_small = vis_bspec_and_calib_in_grid(bev_small, bspec)

        cv2.imshow("img_small", img_small)
        cv2.imshow("bev_small", bev_small)
        cv2.waitKey(0)
        
    video.close()