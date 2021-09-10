import argparse
import bev
import cv2
import numpy as np
import os

from bev.constructor.homo_constr import load_calib, preset_bspec
from bev.io.utils import video_generator
from bev.visualizer.homo_vis import vis_bspec_and_calib_in_grid
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-tag", type=str, default="5_left")
    parser.add_argument("--video-path", type=str,
        default="/media/sda1/datasets/extracted/BrnoCompSpeed/dataset/session{video_tag}/video.avi")
    parser.add_argument("--calib-path", type=str,
        default="/media/sda1/datasets/extracted/BrnoCompSpeed/calib/session{video_tag}/system_dubska_optimal_calib.json")
    parser.add_argument("--no-grid", action="store_true", help="skip drawing the homographic grid plane")
    parser.add_argument("--no-show", action="store_true", help="skip showing output frames while processing")
    parser.add_argument("--no-small", action="store_true",
        help="skip generating another bev from lower res source frames")
    parser.add_argument("--write-vid", action="store_true", help="generate a bev and ori video for trafcam_3d")
    parser.add_argument("--video-out-dir", type=str,
        default="/media/sda1/datasets/extracted/bev/BrnoCompSpeed/session{video_tag}")
    parser.add_argument("--calib-new-u", type=int, default=852)
    parser.add_argument("--calib-new-v", type=int, default=480)
    args = parser.parse_args()

    # img_path = "/media/sda1/datasets/extracted/KoPER/added/SK_1_empty_road_bev.png"
    # # img_path = "/media/sda1/datasets/extracted/KoPER/Sequence1a/KAB_SK_1_undist/KAB_SK_1_undist_1384779301359985.bmp"
    # # img_path = "/media/sda1/datasets/extracted/KoPER/Sequence1a/KAB_SK_4_undist/KAB_SK_4_undist_1384779301359985.bmp"
    # img = cv2.imread(img_path)

    # img = vis_bspec_and_calib_in_grid(img, bspec)
    # # img = vis_bspec_and_calib_in_grid(img, bspec, calib)
    # cv2.imshow("img", img)
    # cv2.waitKey()

    video_tag = args.video_tag
    cam_id = int(video_tag[0])
    cam_sub = video_tag.split("_")[1]
    sub_id_dict = {"left":0.1, "center":0.2, "right":0.3}
    cam_id = cam_id + sub_id_dict[cam_sub]
    # print("cam_id", cam_id)

    video_path = args.video_path.format(video_tag=video_tag) if args.video_path.find("{video_tag}") > -1 else args.video_path
    calib_path = args.calib_path.format(video_tag=video_tag) if args.calib_path.find("{video_tag}") > -1 else args.calib_path
    no_grid = args.no_grid
    no_show = args.no_show
    no_small = args.no_small or args.no_show
    write_vid = args.write_vid
    video_out_dir = args.video_out_dir.format(video_tag=video_tag) if args.video_out_dir.find("{video_tag}") > -1 else args.video_out_dir
    video_out_bev_path = os.path.join(video_out_dir, "bev.mp4")
    video_out_ori_path = os.path.join(video_out_dir, "ori.mp4")

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

    new_u = args.calib_new_u
    new_v = args.calib_new_v
    calib_small = calib.scale(align_corners=False, new_u=new_u, new_v=new_v)
    center_world_small = calib_small.gen_center_in_world()
    print("center_world_small", center_world_small)

    H_world_img_small = calib_small.gen_H_world_img()
    H_bev_img_small = np.linalg.inv(H_world_bev).dot(H_world_img_small)

    video_out_bev = video_generator(video_out_bev_path, width=bspec.u_size, height=bspec.v_size) if write_vid else None
    ori_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out_ori = video_generator(video_out_ori_path, width=ori_width, height=ori_height) if write_vid else None

    while video.isOpened():
        read_result, img = video.read()
        if not read_result: break

        bev = cv2.warpPerspective(img, H_bev_img, (bspec.u_size, bspec.v_size))
        img_small = None if no_small else cv2.resize(img, (new_u, new_v))
        bev_small = None if no_small else cv2.warpPerspective(img_small, H_bev_img_small, (bspec.u_size, bspec.v_size))

        if not no_grid:
            img_small = None if no_small else vis_bspec_and_calib_in_grid(img_small, bspec, calib)
            bev_small = None if no_small else vis_bspec_and_calib_in_grid(bev_small, bspec)
            img = vis_bspec_and_calib_in_grid(img, bspec, calib)
            bev = vis_bspec_and_calib_in_grid(bev, bspec)

        if not no_show:
            cv2.imshow("img", img)
            cv2.imshow("bev", bev)

            if not no_small:
                cv2.imshow("img_small", img_small)
                cv2.imshow("bev_small", bev_small)
            
            cv2.waitKey(0)

        if write_vid:
            video_out_ori.write(img)
            video_out_bev.write(bev)

    video.release()

    if write_vid:
        video_out_ori.release()
        video_out_bev.release()
