from ..tracker.rbox_tracker import Sort
from ..io.rbox_io import read_txt_yolo_pred
from ..io.utils import video_parser, video_generator
from ..constructor.homo_constr import preset_bspec
from ..rbox import rbox_world_bev, xy82xywhr
from ..visualizer.rbox_vis import vis_rbox

import cv2
import os
import numpy as np

import argparse

if __name__ == "__main__":
    fps = 15
    # tracker = Sort(mode="bicycle", max_age=60, min_hits_init=10, min_hits_recover=2, iou_threshold=0.3, fps=60) 
    # tracker = Sort(mode="bicycle", max_age=30, min_hits_init=5, min_hits_recover=2, iou_threshold=0.3, fps=20) 
    tracker = Sort(mode="bicycle", max_age=20, min_hits_init=5, min_hits_recover=2, iou_threshold=0.01, fps=fps, iou_threshold_suppress=-4) #max_age=30
    # This fps corresponds to number of frames per "real-world second", which may not be the same as the fps in video spec. original video use fps=60

    # bspec = preset_bspec("KoPER", 1)
    # bspec = preset_bspec("lturn")
    bspec = preset_bspec("roundabout")

    H_world_bev = bspec.gen_H_world_bev()
    H_bev_world = np.linalg.inv(H_world_bev)

    # video_path = "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/video.avi"
    # pred_path = "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/outputs/KoPER_all_video2/video.txt"
    # video_out_path = "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/outputs/KoPER_all_video2/tracking.avi"
    # dets_xywhr, frame_ids = read_txt_yolo_pred(pred_path, "xywhr")
    # out_video = video_generator(video_out_path, 416, 544)

    # video_path = "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46-bev3000.mkv"
    # pred_path = "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46-1300/yolo-bev20/2020-05-19 16-49-46-bev3000.txt"
    # video_out_path = "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46-1300/yolo-bev20/tracking.mkv"

    # dets, frame_ids = read_txt_yolo_pred(pred_path, "xy8")
    # out_video = video_generator(video_out_path, 544, 416)
    # dets_xywhr = xy82xywhr(dets, "bev")


    # video_path = "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/07/videos/bev30000.mkv"
    # pred_path = "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/07/outputs_gen_CARLA_half_angle_shadow/videos/bev30000.txt"
    # video_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/07/outputs_gen_CARLA_half_angle_shadow/videos/bev30000_tracking2.mkv"
    # txt_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_roundabout/07/outputs_gen_CARLA_half_angle_shadow/videos/bev30000_tracking2.txt"

    # video_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev3000.mkv"
    # pred_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_180_dv__l_C_K_KA2000/bev3000.txt"
    # video_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_180_dv__l_C_K_KA2000/bev3000_tracking.mkv"
    # txt_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_180_dv__l_C_K_KA2000/bev3000_tracking.txt"

    # video_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/videos/bev30000.mkv"
    # pred_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev30000.txt"
    # video_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev30000_tracking.mkv"
    # txt_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/00/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev30000_tracking.txt"

    # video_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/videos/bev_10min_3x.mkv"
    # pred_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev_10min_3x.txt"
    # video_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev_10min_3x_tracking.mkv"
    # txt_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev_10min_3x_tracking.txt"

    # video_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/videos/bev_3x.mkv"
    # pred_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev_3x.txt"
    # video_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev_3x_tracking.mkv"
    # txt_out_path = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_360_dv__l_C_K_KA2000/bev_3x_tracking.txt"

    # video_root = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/videos"
    # pred_root = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/SonganProvided/02/outputs/last_ra_tail_180_dv__l_C_K_KA2000_correct_texture"
    # output_root = pred_root
    # video_name = "bev_3x.mkv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=str)
    parser.add_argument("--pred-root", type=str)
    parser.add_argument("--video-name", type=str)

    args = parser.parse_args()
    video_root = args.video_root
    pred_root = args.pred_root
    video_name = args.video_name

    # video_root = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/videos/sun/"
    # pred_root = "/media/sda1/datasets/extracted/roadmanship_format/test_lturn/MinghanCollected/outputs/best_ra_tail_180_dv__l_C_K_KA2000_correct_texturedq/sun"
    output_root = pred_root
    # video_name = "bev_2020-06-30 17-43-28_1hr.mkv"

    video_path = os.path.join(video_root, video_name)
    pred_path = os.path.join(pred_root, video_name[:-4]+".txt")
    video_out_path = os.path.join(output_root, video_name[:-4] + "_tracking.mkv")
    txt_out_path = os.path.join(output_root, video_name[:-4] + "_tracking.txt")

    print("video_out_path:", video_out_path)

    dets_xywhr, frame_ids = read_txt_yolo_pred(pred_path, "xywhr")
    out_video = video_generator(video_out_path, dim_from=video_path)
    # out_video = video_generator(video_out_path, 544, 544)

    out_txt = open(txt_out_path, "w")
    out_txt.write(str(fps)+"\n")

    for frame, i in video_parser(video_path):
        ### transform bev detections to world
        detections = dets_xywhr[frame_ids==i]
        detections_world = rbox_world_bev(detections, H_world_bev, "bev")

        ### track objects in world
        trackings_world, remaining_world = tracker.update(detections_world)

        ### visualize tracked objects in bev
        rboxes_tracked_world = trackings_world[:, :5]
        rbox_tracked_bev = rbox_world_bev(rboxes_tracked_world, H_bev_world, "world")
        txts = [str(int(x)) for x in trackings_world[:,-1]]
        # v_str_tracked = ["%.1f"%x for x in trackings_world[:,5]] # visualize speed in tracking result video

        rboxes_remaining_world = remaining_world[:, :5]
        rbox_remaining_bev = rbox_world_bev(rboxes_remaining_world, H_bev_world, "world")
        txts_remaining = [str(int(x)) for x in remaining_world[:,-1]]
        # v_str_remaining = ["%.1f"%x for x in remaining_world[:,5]] # visualize speed in tracking result video

        frame = vis_rbox(frame, rbox_remaining_bev, txts=txts_remaining, rbox_color=(0,255,255), txt_color=(0,0,255)) # txts=txts_remaining
        
        frame = vis_rbox(frame, detections, rbox_color=(0,255,0))
        frame = vis_rbox(frame, rbox_tracked_bev, txts=txts, rbox_color=(0,255,255), txt_color=(0,0,255), rbox_fill=True) # txts=txts
        cv2.imshow("frame_vis", frame)
        cv2.waitKey(1)

        out_video.write(frame)

        ### log all tracked objects to a single txt file
        for j in range(trackings_world.shape[0]):
            vid = int(trackings_world[j, -1])
            fid = i
            obs_flag = 1
            out_txt.write("{} {} {} {}\n".format(vid, fid, obs_flag, " ".join("{:.4f}".format(x) for x in trackings_world[j, :7]) ))

        for j in range(remaining_world.shape[0]):
            vid = int(remaining_world[j, -1])
            fid = i
            obs_flag = 0
            out_txt.write("{} {} {} {}\n".format(vid, fid, obs_flag, " ".join("{:.4f}".format(x) for x in remaining_world[j, :7]) ))
        

    out_video.release()

    out_txt.close()