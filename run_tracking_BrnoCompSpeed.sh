#!/bin/bash

video_name=(
    # "4_center"
    # "4_right"
    # "5_left"
    # "5_center"
    # "5_right"
    # "6_left"

    "6_center"
    # "6_right"
    "4_left"
)

cd ..

for i in ${!video_name[*]}; do 
    echo "video_name: "${video_name[$i]}

    python -m bev.tool.rbox_tracking_BrnoCompSpeed \
    --video-root "/media/sda1/datasets/extracted/roadmanship_format/BrnoCompSpeed/" \
    --video-tag ${video_name[$i]} --fps 50 --no-vid --no-vis
done