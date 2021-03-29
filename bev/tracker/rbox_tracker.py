# """Minghan: tracker of rbox, modifid from https://github.com/abewley/sort
# What need change: box format, iou calculation, model in Kalman filter
# requirements: 
# filterpy==1.4.5
# scikit-image==0.14.0
# lap==0.4.0
# Notice that calculating rbox iou requires torch tensor input. Can also use lin_iou which is simply center offset ratio
# Do the tracking in real world space, where the scale is consistent and do not need to tune across different BEV images. 
# """
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter ### ExtendedKalmanFilter may not be used. 

try: 
    import d3d
    import torch
except ImportError:
    print("install d3d from https://github.com/cmpute/d3d")
    raise ImportError

from ..evaluator.kpts_eval import lin_iou, lin_iou_ellipsoid

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# def iou_batch(bb_test, bb_gt):                                                                                                   
#   """                                                                                                                      
#   From SORT: Computes IUO between two bboxes in the form [l,t,w,h]                                                         
#   """                                                                                                                      
#   bb_gt = np.expand_dims(bb_gt, 0)                                                                                         
#   bb_test = np.expand_dims(bb_test, 1)                                                                                     
                                                                                                                           
#   xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])                                                                         
#   yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])                                                                         
#   xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])                                                                         
#   yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])                                                                         
#   w = np.maximum(0., xx2 - xx1)                                                                                            
#   h = np.maximum(0., yy2 - yy1)                                                                                            
#   wh = w * h                                                                                                               
#   o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
#   return(o)  

def iou_batch(bb_test, bb_gt):
  """for rbox use lin iou to begin with"""
  return lin_iou(bb_test, bb_gt, bb_gt)

def iou_batch_rbox(bb_test, bb_gt):
  bb_test_ = bb_test.copy()
  bb_test_[:, 4] = - bb_test[:, 4]
  bb_gt_ = bb_gt[:, :5].copy()
  bb_gt_[:, 4] = - bb_gt[:, 4]
  return d3d.box.box2d_iou(bb_test_, bb_gt_, method="rbox")

def iou_batch_ellipsoid(bb_test, bb_gt):
  """for rbox use lin iou to begin with"""
  return lin_iou_ellipsoid(bb_test, bb_gt, bb_gt)

def convert_bbox_to_z(bbox):
  """from detection[x,y,w,h,r,score] to 5-dim rbox observation z=[x,y,w,h,r]"""
  return bbox[:5].reshape(5,1)

def convert_x_to_bbox(x,score=None):
  """for rbox, from 7-dim state [x,y,w,h,r,us,ur] to rbox [[x,y,w,h,r]]"""
  return x[:5].reshape(1,5)

def convert_x_to_bbox_vr(x,score=None):
  """for rbox, from 7-dim state [x,y,w,h,r,us,ur] to rbox [[x,y,w,h,r]]"""
  return x[:7].reshape(1,7)

### modified from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb
def f_state_trans(x, dt):
  """state transition function: X_{k+1}=f(x_{k}). There is no u input here. 
  X = [x, y, w(width), h(length), r(yaw angle), us (linear velocity), ur (angular velocity)]
  [ x_{k+1}  = x_k + us_k * cos(r_k) * dt
    y_{k+1}  = y_k + us_k * sin(r_k) * dt
    w_{k+1}  = w_k
    h_{k+1}  = h_k
    r_{k+1}  = r_k + ur_k * dt
    us_{k+1} = us_k
    ur_{k+1} = ur_k ]
    The coordinates are defined in world coordinate, independent to BEV parameters and dataset-specific coordinate
  """
  dx = np.zeros_like(x)   # 7*1
  x_, y_, w, h, r, us, ur = x
  dx[0] = us * np.cos(r) * dt
  dx[1] = us * np.sin(r) * dt
  dx[4] = ur * dt
  x_new = x + dx
  return x_new

def f_state_trans_bi(x, dt, rlfr=0.4):
  """state transition function: X_{k+1}=f(x_{k}) with a bicycle model. There is no u input here. 
  X = [x, y, w(width), h(length), r(yaw angle), v (linear velocity), a (front wheel steering angle)]
  lf = rlfr * l
  lr = rlfr * r (we assume lf = lr)
  b = np.atan(lr / (lr + lf) * tan(a))
  [ x_{k+1}  = x_k + v_k * cos(r_k + b) * dt
    y_{k+1}  = y_k + v_k * sin(r_k + b) * dt
    w_{k+1}  = w_k
    h_{k+1}  = h_k
    r_{k+1}  = r_k + v_k / lr * sin(b) * dt
    v_{k+1} = v_k
    a_{k+1} = a_k ]
  The coordinates are defined in world coordinate, independent to BEV parameters and dataset-specific coordinate. 
  This bicycle model refers to: 
  "Kinematic and dynamic vehicle models for autonomous driving control design" http://ieeexplore.ieee.org/document/7225830/
  """
  x_, y_, w, h, r, v, a = x
  lf = rlfr * h
  lr = rlfr * h
  b = np.arctan(0.5 * np.tan(a))
  dbda = 1 / (1+ (0.5 * np.tan(a))**2) * 0.5 / (np.cos(a)**2)

  dx = np.zeros_like(x)   # 7*1
  dx[0] = v * np.cos(r + b) * dt
  dx[1] = v * np.sin(r + b) * dt
  dx[4] = v / lr * np.sin(b) * dt

  x_new = x + dx
  return x_new

def FJacobian_bi(x, dt, rlfr=0.4):
  """Jacobian of state transition function. """
  x_, y_, w, h, r, v, a = x
  lf = rlfr * h
  lr = rlfr * h
  b = np.arctan(0.5 * np.tan(a))
  dbda = 1 / (1+ (0.5 * np.tan(a))**2) * 0.5 / (np.cos(a)**2)

  FJ = np.eye(7)
  FJ[0, 4] = -v * np.sin(r + b) * dt
  FJ[0, 5] = np.cos(r + b) * dt
  FJ[0, 6] = -v * np.sin(r + b) * dt * dbda
  FJ[1, 4] = v * np.cos(r + b) * dt
  FJ[1, 5] = np.sin(r + b) * dt
  FJ[1, 6] = v * np.cos(r + b) * dt * dbda
  FJ[4, 5] = 1 / lr * np.sin(b) * dt
  FJ[4, 6] = v / lr * np.cos(b) * dt * dbda
  return FJ

def FJacobian(x, dt):
  """Jacobian of state transition function. """
  x_, y_, w, h, r, us, ur = x
  FJ = np.eye(7)
  FJ[0, 4] = -us * np.sin(r) * dt
  FJ[0, 5] = np.cos(r) * dt
  FJ[1, 4] = us * np.cos(r) * dt
  FJ[1, 5] = np.sin(r) * dt
  FJ[4, 6] = dt
  return FJ

def angle_residual(a, b):
  """ compute residual (a-b) between two angles. """
  y = a - b
  ### if we want to distinguish heads and tails
  # y = y % (2 * np.pi)    # force in range [0, 2 pi)
  # if y > np.pi:             # move to [-pi, pi)
  #     y -= 2 * np.pi
  ### if we do not want to distinguish heads and tails
  y = y % np.pi    # force in range [0, pi)
  if y > 0.5*np.pi:             # move to [-0.5pi, 0.5pi)
    y -= np.pi
  return y

def rbox_residual(bbox_pred, bbox_gt):
  y = bbox_pred - bbox_gt
  y[4] = angle_residual(bbox_pred[4], bbox_gt[4])
  return y

class EKFNonlinST(ExtendedKalmanFilter):
  """The ExtendedKalmanFilter provided by filterpy does not support nonlinear state transition model, 
  therefore override its predict_x function to support it. """
  def __init__(self, f_state_trans, FJacobian):
    ExtendedKalmanFilter.__init__(self, dim_x=7, dim_z=5)
    self.f_state_trans  = f_state_trans
    self.FJacobian = FJacobian

  def predict_x(self, u):
    self.F = self.FJacobian(self.x)
    self.x = self.f_state_trans(self.x)

# class RboxTracker:
#     def __init__(self, rbox):
#         """rbox is the first rbox detection xywhr"""
#         self.dt = 1/30
#         self.FJacobian = lambda x: FJacobian(x, self.dt)            ### Jacobian of state transition function
#         self.f_state_trans = lambda x: f_state_trans(x, self.dt)    ### state transition function
#         self.EKF = EKFNonlinST(self.f_state_trans, self.FJacobian)
#         self.HJacobian = lambda x: np.eye(5, 7)                     ### Jacobian of observation model
#         self.Hx = lambda x: x[:5]                                   ### observation model
#         R_vec = np.array([1, 1, 0.5, 0.5, 0.3])
#         self.EKF.R = np.diag(R_vec)                                 ### measurement noise
#         Q_vec = np.array([0.5, 0.5, 0.2, 0.2, 0.1, 1, 0.1])
#         self.EKF.Q = np.diag(Q_vec)                                 ### Process noise 

#         ### initialize the EKF
#         x0 = np.zeros((7,1))
#         x0[:5] = rbox
#         x0[5] = 5
#         self.EKF.x = x0
#         self.EKF.F = self.FJacobian(self.EKF.x)
#         P0_vec = np.array([2, 2, 1, 1, 0.5, 5, 0.5])
#         self.EKF.P = np.diag(P0_vec)


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox, mode, min_hits_init=3, min_hits_recover=3, fps=60):
  # def __init__(self,rbox):
    """
    Initialises a tracker using initial bounding box.
    mode in ["unicycle", "bicycle"]
    """
    #################################################### rbox
    #### define the Kalman filter for rbox tracking
    if mode == "unicycle":
      self.dt = 1.0/fps
      self.FJacobian = lambda x: FJacobian(x, self.dt)            ### Jacobian of state transition function
      self.f_state_trans = lambda x: f_state_trans(x, self.dt)    ### state transition function
      self.kf = EKFNonlinST(self.f_state_trans, self.FJacobian)
      self.HJacobian = lambda x: np.eye(5, 7)                     ### Jacobian of observation model
      self.Hx = lambda x: x[:5]                                   ### observation model: Z = X[:5] (Z=[x,y,w,h,r])
      self.residual = rbox_residual                               ### compute the residual, needed when the residual computation is nonlinear
      R_vec = np.array([1, 1, 0.5, 0.5, 0.3])
      self.kf.R = np.diag(R_vec)                                 ### measurement noise
      # Q_vec = np.array([0.5, 0.5, 0.2, 0.2, 0.1, 1, 0.1])
      Q_vec = np.array([0.5, 0.5, 0.2, 0.2, 0.01, 1, 0.1]) * 0.5
      Q_vec = Q_vec * self.dt
      self.kf.Q = np.diag(Q_vec)                                 ### Process noise 

      ### initialize the kf
      x0 = np.zeros((7,1))
      x0[:5] = convert_bbox_to_z(bbox)
      x0[5] = 5
      self.kf.x = x0
      self.kf.F = self.FJacobian(self.kf.x)                       ### Jacobian of state transition function
      P0_vec = np.array([2, 2, 1, 1, 0.5, 100000, 0.5])
      self.kf.P = np.diag(P0_vec)                                 ### state uncertainty

    elif mode == "bicycle":
      self.dt = 1.0/fps
      self.FJacobian = lambda x: FJacobian_bi(x, self.dt)            ### Jacobian of state transition function
      self.f_state_trans = lambda x: f_state_trans_bi(x, self.dt)    ### state transition function
      self.kf = EKFNonlinST(self.f_state_trans, self.FJacobian)
      self.HJacobian = lambda x: np.eye(5, 7)                     ### Jacobian of observation model
      self.Hx = lambda x: x[:5]                                   ### observation model: Z = X[:5] (Z=[x,y,w,h,r])
      self.residual = rbox_residual                               ### compute the residual, needed when the residual computation is nonlinear
      R_vec = np.array([2, 2, 0.5, 0.5, 1])
      self.kf.R = np.diag(R_vec)                                 ### measurement noise
      Q_vec = np.array([0.5, 0.5, 0.2, 0.2, 0.01, 1, 0.01]) * 0.5
      Q_vec = Q_vec * self.dt
      self.kf.Q = np.diag(Q_vec)                                 ### Process noise 

      ### initialize the kf
      x0 = np.zeros((7,1))
      x0[:5] = convert_bbox_to_z(bbox)
      x0[5] = 0
      self.kf.x = x0
      self.kf.F = self.FJacobian(self.kf.x)                       ### Jacobian of state transition function
      P0_vec = np.array([2, 2, 1, 1, 0.5, 100000, 0.5])
      self.kf.P = np.diag(P0_vec)                                 ### state uncertainty


    #################################################### 2d bbox
    # #define constant velocity model
    # self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    # self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    # self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    # self.kf.R[2:,2:] *= 10.
    # self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # self.kf.P *= 10.
    # self.kf.Q[-1,-1] *= 0.01
    # self.kf.Q[4:,4:] *= 0.01

    # self.kf.x[:4] = convert_bbox_to_z(bbox)
    ##############################################################

    self.time_since_update = 0        ### the number of continuous predictions when observations are missing
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []         ### continuous predictions when observations are missing
    self.hits = 0             ### the number of observations
    self.hit_streak = 0       ### the number of continuous observations
    self.age = 0

    ##############################################################
    ### parameters for managing the activity of the tracked object
    self.min_hits_init = min_hits_init
    self.min_hits_recover = min_hits_recover
    self.validated = False    ### indicating whether the tracker is ever validated
    self.active = False

  def validate(self):
    if not self.validated:
      if self.hit_streak > self.min_hits_init:
        self.validated = True
    
    if self.time_since_update == 0 and self.hit_streak >= self.min_hits_recover and self.validate:
      ### self.time_since_update == 0 means there is an observation in current step
      ### self.hit_streak >= self.min_hits_recover means there are already some continuous observation
      ### self.validate == True means it has passed the initialization stage. 
      self.active = True
    else:
      self.active = False

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    # self.kf.update(convert_bbox_to_z(bbox))
    self.kf.update(convert_bbox_to_z(bbox), HJacobian=self.HJacobian, Hx=self.Hx, residual=self.residual)   # for EKF

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    # if((self.kf.x[6]+self.kf.x[2])<=0):
    #   self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    # return convert_x_to_bbox(self.kf.x)
    return convert_x_to_bbox_vr(self.kf.x)    ## include v and steering angle output


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3, mode='rbox_iou', non_exclusive=False):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0) or (len(detections)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,),dtype=int), detections, trackers

  assert mode in ['rbox_iou', 'lin_iou', 'ellipsoid_iou']
  if mode == 'rbox_iou':
    iou_matrix = iou_batch_rbox(detections, trackers)
  elif mode == 'lin_iou':
    iou_matrix = iou_batch(detections, trackers)
  elif mode == 'ellipsoid_iou':
    iou_matrix = iou_batch_ellipsoid(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if (a.sum(1).max() == 1 and a.sum(0).max() == 1) or non_exclusive:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  unmatched_detections_item = [detections[i] for i in unmatched_detections]
  unmatched_trackers_item = [trackers[i] for i in unmatched_trackers]
  unmatched_detections_item = np.array(unmatched_detections_item)
  unmatched_trackers_item = np.array(unmatched_trackers_item)
  # print("unmatched_detections_item", unmatched_detections_item.shape)
  # print("unmatched_trackers_item", unmatched_trackers_item.shape)
  
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers), unmatched_detections_item, unmatched_trackers_item


class Sort(object):
  def __init__(self, mode, max_age=1, min_hits_init=3, min_hits_recover=3, iou_threshold=0.3, fps=60, iou_threshold_suppress=-1):
    """
    Sets key parameters for SORT
    mode in ["unicycle", "bicycle"]
    """
    self.mode = mode
    self.max_age = max_age
    self.min_hits_init = min_hits_init
    self.min_hits_recover = min_hits_recover
    self.iou_threshold = iou_threshold
    self.iou_threshold_suppress = iou_threshold_suppress
    self.fps = fps
    self.trackers = []
    self.trackers_init = []
    self.frame_count = 0

    ### Minghan: to be compatible with rbox
    # self.boxdim = 4 ### for 2d bbox
    self.boxdim = 5 ### for rbox

#   def update(self, dets=np.empty((0, 5))):
  def update(self, dets=None):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    if dets is None:
      dets = np.empty((0, self.boxdim+1))
    self.frame_count += 1
    # get predicted locations from existing trackers.
    # trks = np.zeros((len(self.trackers), 5))
    trks = np.zeros((len(self.trackers), self.boxdim+1))
    to_del = []
    ret = []
    remaining = []        ### tracked but not observed objects
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
    #   trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      trk[:] = [*pos, 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks, unmatched_dets_item, unmatched_trks_item = associate_detections_to_trackers(dets, trks, self.iou_threshold, mode='rbox_iou') # rbox_iou

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    trks = np.zeros((len(self.trackers_init), self.boxdim+1))
    to_del = []
    for t, trk in enumerate(trks):
      pos = self.trackers_init[t].predict()[0]
    #   trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      trk[:] = [*pos, 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers_init.pop(t)
    
    matched_init, unmatched_dets_init, unmatched_trks_init, unmatched_dets_init_item, unmatched_trks_init_item = \
      associate_detections_to_trackers(unmatched_dets_item, trks, self.iou_threshold, mode='rbox_iou') # rbox_iou

    # update matched trackers with assigned detections
    for m in matched_init:
      self.trackers_init[m[1]].update(unmatched_dets_item[m[0], :])

    # matched_suppress, unmatched_dets_suppress, unmatched_trks_suppress, _, _ = associate_detections_to_trackers(unmatched_dets_init_item, unmatched_trks_item, self.iou_threshold_suppress, mode='lin_iou', non_exclusive=True)


    # create and initialise new trackers for unmatched detections
    for ii in unmatched_dets_init:
        i = unmatched_dets[ii]
    # for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:], mode=self.mode, min_hits_init=self.min_hits_init, min_hits_recover=self.min_hits_recover, fps=self.fps)
        # self.trackers.append(trk)
        self.trackers_init.append(trk)
    
    i = len(self.trackers_init)
    for trk in reversed(self.trackers_init):
        trk.validate()
        d = trk.get_state()[0]
        i -= 1
        if trk.time_since_update > self.max_age:
          self.trackers_init.pop(i)
          continue
        if trk.validated:
          self.trackers_init.pop(i)
          self.trackers.append(trk)
          continue
        if trk.time_since_update < 1 and self.frame_count <= self.min_hits_init:
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        trk.validate()
        d = trk.get_state()[0]
        i -= 1
        if trk.active or (trk.time_since_update < 1 and self.frame_count <= self.min_hits_init):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        # remove dead tracklet
        elif trk.time_since_update > self.max_age:
          self.trackers.pop(i)
        elif trk.validated:
          remaining.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))


    if(len(ret)>0):
      ret = np.concatenate(ret)
    else:
      ret = np.empty((0,8)) # 5

    if(len(remaining)>0):
      remaining = np.concatenate(remaining)
    else:
      remaining = np.empty((0,8)) # 5
    
    return ret, remaining       ### changed to output both observed and unobserved objects

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--fps", 
                        help="frame rate of input video", 
                        type=int, default=60)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold, 
                       fps=args.fps ) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split('/')[0]
    
    with open('output/%s.txt'%(seq),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase, seq, frame)
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers, _ = mot_tracker.update(dets)      ### ignore the unobserved objects
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
