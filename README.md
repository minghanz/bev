# BEV
This repo is a utility package for bird's eye vire related functions. 

Examples include: 
- Configuring parameters for homography transform from camera parameters and hand-picked parameters. 
- Visualizing the homography on original view and BEV. 
- Tracking and visualizing rotated bounding boxes in BEV. 

It is used by the paper [**Monocular 3D Vehicle Detection Using Uncalibrated Traffic Cameras through Homography**](https://arxiv.org/pdf/2103.15293.pdf). 

The tracking part of the code requires calculating iou between rotated bounding boxes, requiring the [d3d](https://github.com/cmpute/d3d) package. Otherwise you do not need the d3d package. 

Use the package by running: 
```
python setup.py develop
```