from ..tracker.rbox_tracker import Sort
from ..io.rbox_io import read_txt_yolo_pred
from ..io.utils import video_parser, video_generator
from ..constructor.homo_constr import preset_bspec, load_calib
from ..rbox import rbox_world_bev, xy82xywhr, rbox_world_img
from ..visualizer.rbox_vis import vis_rbox

import cv2
import os
import numpy as np

import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=str)

    parser.add_argument("--video-tag", type=str)
    parser.add_argument("--fps", type=int, help="how many frames are recorded in one actual second")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--no-vid", action="store_true")

    args = parser.parse_args()
    video_root = args.video_root
    video_tag = args.video_tag
    fps = args.fps
    no_vis = args.no_vis
    no_vid = args.no_vid

    video_path = os.path.join(video_root, "session"+video_tag, "videos", "bev.avi")
    pred_path = os.path.join(video_root, "session"+video_tag, "outputs", "bev.txt")
    video_out_path = os.path.join(video_root, "session"+video_tag, "outputs", "bev_tracking.avi")
    txt_out_path = os.path.join(video_root, "session"+video_tag, "outputs", "bev_tracking.txt")

    calib_path = os.path.join(video_root, "session"+video_tag, "calibs", "system_dubska_optimal_calib.json")
    calib_src_path = os.path.join("/media/sda1/datasets/extracted/BrnoCompSpeed/results", "session"+video_tag, "system_dubska_optimal_calib.json")
    
    video_ori_path = os.path.join(video_root, "session"+video_tag, "videos", "ori.avi")
    video_out_ori_path = os.path.join(video_root, "session"+video_tag, "outputs", "ori_tracking.avi")
    json_out_path = os.path.join(video_root, "session"+video_tag, "outputs", "system_ori_tracking.json")

    print("video_out_path:", video_out_path)

    cam_id = int(video_tag[0])
    cam_sub = video_tag.split("_")[1]
    sub_id_dict = {"left":0.1, "center":0.2, "right":0.3}
    cam_id = cam_id + sub_id_dict[cam_sub]

    # tracker = Sort(mode="bicycle", max_age=60, min_hits_init=10, min_hits_recover=2, iou_threshold=0.3, fps=60) 
    # tracker = Sort(mode="bicycle", max_age=30, min_hits_init=5, min_hits_recover=2, iou_threshold=0.3, fps=20) 
    tracker = Sort(mode="bicycle", max_age=fps, min_hits_init=int(0.2*fps), min_hits_recover=int(0.1*fps), iou_threshold=0.3, fps=fps) 
    # This fps corresponds to number of frames per "real-world second", which may not be the same as the fps in video spec. original video use fps=60

    calib = load_calib("BrnoCompSpeed", calib_path)
    calib_small = calib.scale(align_corners=False, new_u=852, new_v=480)
    bspec = preset_bspec("BrnoCompSpeed", cam_id, calib_small)

    H_world_bev = bspec.gen_H_world_bev()
    H_bev_world = np.linalg.inv(H_world_bev)

    H_world_img = calib.gen_H_world_img()
    H_img_world = np.linalg.inv(H_world_img)

    H_world_img_small = calib_small.gen_H_world_img()
    H_img_world_small = np.linalg.inv(H_world_img_small)

    dets_xywhr, frame_ids = read_txt_yolo_pred(pred_path, "xywhr")
    if not no_vid:
        out_video = video_generator(video_out_path, dim_from=video_path)
        out_video_ori = video_generator(video_out_ori_path, dim_from=video_ori_path)
        # out_video = video_generator(video_out_path, 544, 544)

    out_txt = open(txt_out_path, "w")
    out_txt.write(str(fps)+"\n")

    vehs = dict()

    out_dict = dict()

    with open(calib_src_path) as f:
        data = json.load(f)
        out_dict["camera_calibration"] = data["camera_calibration"]

    for (frame, i), (frame_ori, _) in zip(video_parser(video_path), video_parser(video_ori_path)):
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

        xys_tracked_img = rbox_world_img(rboxes_tracked_world, H_img_world)
        xys_tracked_img_small = rbox_world_img(rboxes_tracked_world, H_img_world_small)

        xys_remaining_img = rbox_world_img(rboxes_remaining_world, H_img_world)
        xys_remaining_img_small = rbox_world_img(rboxes_remaining_world, H_img_world_small)

        if not no_vis:
            frame = vis_rbox(frame, rbox_remaining_bev, txts=txts_remaining, rbox_color=(0,255,255), txt_color=(0,0,255)) # txts=txts_remaining
            
            frame = vis_rbox(frame, detections, rbox_color=(0,255,0))
            frame = vis_rbox(frame, rbox_tracked_bev, txts=txts, rbox_color=(0,255,255), txt_color=(0,0,255), rbox_fill=True) # txts=txts
            
            for xy_img_small in xys_remaining_img_small:
                frame_ori = cv2.circle(frame_ori, (int(xy_img_small[0]), int(xy_img_small[1])), 5, (0,255,255) )

            for xy_img_small in xys_tracked_img_small:
                frame_ori = cv2.circle(frame_ori, (int(xy_img_small[0]), int(xy_img_small[1])), 5, (0,255,0), -1 )

            cv2.imshow("frame_bev", frame)
            cv2.imshow("frame_ori", frame_ori)
            cv2.waitKey(1)

            if not no_vid:
                out_video.write(frame)
                out_video_ori.write(frame_ori)

        ### log all tracked objects to a single txt file
        for j in range(trackings_world.shape[0]):
            vid = int(trackings_world[j, -1])
            fid = i
            obs_flag = 1
            out_txt.write("{} {} {} {}\n".format(vid, fid, obs_flag, " ".join("{:.4f}".format(x) for x in trackings_world[j, :7]) ))

            if vid not in vehs:
                vehs[vid] = dict()
                vehs[vid]["id"] = vid
                vehs[vid]["frames"] = []
                vehs[vid]["posX"] = []
                vehs[vid]["posY"] = []
            vehs[vid]["frames"].append(fid)
            vehs[vid]["posX"].append(xys_tracked_img[j,0])
            vehs[vid]["posY"].append(xys_tracked_img[j,1])
                

        for j in range(remaining_world.shape[0]):
            vid = int(remaining_world[j, -1])
            fid = i
            obs_flag = 0
            out_txt.write("{} {} {} {}\n".format(vid, fid, obs_flag, " ".join("{:.4f}".format(x) for x in remaining_world[j, :7]) ))
        
            if vid not in vehs:
                vehs[vid] = dict()
                vehs[vid]["id"] = vid
                vehs[vid]["frames"] = []
                vehs[vid]["posX"] = []
                vehs[vid]["posY"] = []
            vehs[vid]["frames"].append(fid)
            vehs[vid]["posX"].append(xys_remaining_img[j,0])
            vehs[vid]["posY"].append(xys_remaining_img[j,1])


    if not no_vid:
        out_video.release()
        out_video_ori.close()

    out_txt.close()

    veh_list = []
    for vid in vehs:
        veh_list.append(vehs[vid])

    out_dict["cars"] = veh_list

    with open(json_out_path, "w") as f:
        json.dump(out_dict, f)