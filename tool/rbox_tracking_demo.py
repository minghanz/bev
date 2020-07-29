from ..tracker.rbox_tracker import Sort
from ..io.rbox_io import read_txt_yolo_pred
from ..io.utils import video_parser
from ..constructor.homo_constr import preset_bspec
from ..rbox import rbox_world_bev, xy82xywhr
from ..visualizer.rbox_vis import vis_rbox

import cv2
import os
import numpy as np

if __name__ == "__main__":

    tracker = Sort(max_age=10,min_hits=3, iou_threshold=0.3)
    # bspec = preset_bspec("KoPER", 1)
    bspec = preset_bspec("lturn")
    H_world_bev = bspec.gen_H_world_bev()
    H_bev_world = np.linalg.inv(H_world_bev)

    # video_path = "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/video.avi"
    # pred_path = "/media/sda1/datasets/extracted/KoPER_BEV/Sequence1a/KAB_SK_1_undist/outputs/KoPER_all_video2/video.txt"

    video_path = "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46-bev3000.mkv"
    pred_path = "/media/minghanz/Seagate_Backup_Plus_Drive/ADAS_data/2020-05-19 16-49-46-1300/yolo-bev20/2020-05-19 16-49-46-bev3000.txt"

    # dets, frame_ids = read_txt_yolo_pred(pred_path, "xywhr")

    dets, frame_ids = read_txt_yolo_pred(pred_path, "xy8")
    dets_xywhr = xy82xywhr(dets, "bev")

    for frame, i in video_parser(video_path):
        detections = dets_xywhr[frame_ids==i]
        detections_world = rbox_world_bev(detections, H_world_bev, "bev")

        trackings_world = tracker.update(detections_world)

        rboxes_tracked_world = trackings_world[:, :5]
        rbox_tracked_bev = rbox_world_bev(rboxes_tracked_world, H_bev_world, "world")

        # trackings_bev = np.concatenate([rbox_tracked_bev, trackings_world[:,[-1]]], axis=1)

        txts = [str(int(x)) for x in trackings_world[:,-1]]
        frame_vis = vis_rbox(frame, detections, rbox_color=(255,0,0))
        frame_vis = vis_rbox(frame_vis, rbox_tracked_bev, txts=txts, rbox_color=(0,255,0))
        cv2.imshow("frame_vis", frame_vis)
        cv2.waitKey(0)
