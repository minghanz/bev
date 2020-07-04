import numpy as np
import os

from .frozen_class import FrozenClass
from .homography import homo_from_KRt, homo_from_pts

class Calib(FrozenClass):
    def __init__(self, **kwargs):
        ### intrinsics
        self.K = None#np.zeros((3,3), dtype=np.float32)
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.dist_coeff = None#np.zeros(5, dtype=np.flaot32)
        self.u_size = 0
        self.v_size = 0

        ### extrinsics
        self.R = None#np.zeros((3,3), dtype=np.float32)
        self.t = None#np.zeros((3), dtype=np.float32)
        self.T = None#np.zeros((4,4), dtype=np.float32)

        ### homographics
        self.pts_world = None
        self.pts_image = None

        self.H_world_img = None#np.zeros((3,3), dtype=np.float32)
        self.H_img_world = None#np.zeros((3,3), dtype=np.float32)

        self.mode = None
        self._freeze()
        self.__dict__.update(kwargs)

        if self.pts_image is not None and self.pts_world is not None:
            self.mode = "from_pts"
        else:
            self.mode = "from_KRt"

        self.update()
        self.check_validity()
        
    def update(self):
        rtt_array = [self.K is None, self.R is None, self.t is None, self.T is None]
        
        if all(rtt_array):
            pass
        elif any(rtt_array):
            assert self.K is not None or all(np.array([self.fx, self.fy, self.cx, self.cy])!=None)
            if self.K is None:
                self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0,0,1]], dtype=np.float32)
            if self.T is None and self.R is not None and self.t is not None:
                self.T = np.concatenate(np.concatenate((self.R, self.t.reshape(3,1)), axis=1), np.array([[0,0,0,1]]), axis=0).astype(np.float32)
            elif self.T is not None and self.R is None and self.t is None:
                self.R = self.T[:3,:3]
                self.t = self.T[:3,3]
            else:
                raise ValueError("R,t,T not valid", self.R, self.t, self.T)
        else:
            assert np.allclose(self.T, np.concatenate((np.concatenate((self.R, self.t.reshape(3,1)), axis=1), np.array([[0,0,0,1]])), axis=0).astype(np.float32)), \
                    "{} {} {}".format(self.R, self.t, self.T)
        
    def check_validity(self):
        # rtt_array = np.array([self.K, self.R, self.t, self.T])
        rtt_array = [self.K is None, self.R is None, self.t is None, self.T is None]
        assert all(rtt_array) or not any(rtt_array)
        if not any(rtt_array):
            assert np.allclose(self.T, np.concatenate((np.concatenate((self.R, self.t.reshape(3,1)), axis=1), np.array([[0,0,0,1]])), axis=0).astype(np.float32)), \
                    "{} {} {}".format(self.R, self.t, self.T)
        else:
            assert self.pts_image is not None and self.pts_world is not None


    def gen_H_world_img(self, mode):
        self.check_validity()
        assert mode in ["from_KRt", "from_pts"]

        if mode == "from_pts":
            assert self.pts_image is not None and self.pts_world is not None
            H_world_img = homo_from_pts(self.pts_image, self.pts_world[:,:2])
        else:
            assert self.R is not None and self.t is not None 
            H_img_world = homo_from_KRt(self.K, Rt_homo=self.T)
            H_world_img = np.linalg.inv(H_img_world)

        return H_world_img