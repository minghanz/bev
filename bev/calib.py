import numpy as np
import os

from .frozen_class import FrozenClass
from .homo import homo_from_KRt, homo_from_pts, homo_from_vps

class Calib(FrozenClass):
    def __init__(self, **kwargs):
        """ There are three ways to specify the image-world homography:
        1. K, RT
        2. vanishing points (at least 2) and image dimension
        3. corresponding points between two planes
        Information embedded: 1>2>3. 
        1: determines the full camera geometry. 
        2: determines K, R. The translation of the camera w.r.t. the plane is not determined, which affects the translation (by translation of the camera parallel to the plane) 
        and scaling (by translation of the camera perpendicular to the plane) of the homography. 
        3. Only determines H. The focal length of the camera is not determined, affecting K, R, and T. 
        """
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

        ### vanishing points
        self.vp1 = None #np.zeros((2), dtype=np.float32)
        self.vp2 = None #np.zeros((2), dtype=np.float32)
        self.pp = None
        self.height = None #float
        self.u_size = None #int
        self.v_size = None #int


        self.mode = None
        self._freeze()
        self.__dict__.update(kwargs)

        if self.pts_image is not None and self.pts_world is not None:
            self.mode = "from_pts"
        elif self.vp1 is not None and self.vp2 is not None:
            self.mode = "from_vps"
        else:
            self.mode = "from_KRt"

        self.update()
        self.check_validity()
        
    def update(self):
        """ Fulfill K, R, t, T if any of them is are given. 
        """
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
            
        if self.mode == "from_vps":
            if self.pp is None:
                self.pp = np.zeros_like(self.vp1)
                self.pp[0] = (self.u_size-1) * 0.5
                self.pp[1] = (self.v_size-1) * 0.5
                
        
    def check_validity(self):
        # rtt_array = np.array([self.K, self.R, self.t, self.T])
        rtt_array = [self.K is None, self.R is None, self.t is None, self.T is None]
        assert all(rtt_array) or not any(rtt_array)
        if not any(rtt_array):
            assert np.allclose(self.T, np.concatenate((np.concatenate((self.R, self.t.reshape(3,1)), axis=1), np.array([[0,0,0,1]])), axis=0).astype(np.float32)), \
                    "{} {} {}".format(self.R, self.t, self.T)
        
        if self.mode == "from_pts":
            assert self.pts_image is not None and self.pts_world is not None
        elif self.mode == "from_vps":
            check_array = [self.vp1, self.vp2, self.pp, self.height, self.u_size, self.v_size]
            check_array = [x is not None for x in check_array]
            assert all(check_array)


    def gen_H_world_img(self, mode=None):
        self.check_validity()
        if mode is None:
            mode = self.mode
            
        assert mode in ["from_KRt", "from_pts", "from_vps"], mode

        if mode == "from_pts":
            assert self.pts_image is not None and self.pts_world is not None
            H_world_img = homo_from_pts(self.pts_image, self.pts_world[:,:2])
        elif mode == "from_vps":
            H_img_world = homo_from_vps(self.vp1, self.vp2, self.height, self.u_size, self.v_size, self.pp)
            H_world_img = np.linalg.inv(H_img_world)
        else:
            assert self.R is not None and self.t is not None 
            H_img_world = homo_from_KRt(self.K, Rt_homo=self.T)
            H_world_img = np.linalg.inv(H_img_world)

        return H_world_img

    def gen_center_in_world(self):
        """return the center pixel projected to the road plane (x,y,1), with z=0"""
        H_world_img = self.gen_H_world_img()
        H_img_world = np.linalg.inv(H_world_img)

        # bev
        test_pt = np.array([(self.u_size-1)/2, (self.v_size-1)/2, 1]).reshape(3)
        test_pt_world = H_world_img.dot(test_pt)
        test_pt_world = test_pt_world / test_pt_world[2]
        test_pt_world = test_pt_world.reshape(-1)

        return test_pt_world

    def scale(self, align_corners, new_u=None, new_v=None, scale_ratio_u=None, scale_ratio_v=None):
        """This is to generate a new Calib object based on self object, according to scaling parameters. 
        "align_corners==True" means that the center point of corner pixels are aligned. Otherwise, the corner of corner pixels are aligned. 
        Note that if align_corners==True, scale_ratio_u = (new_width - 1) / (old_width - 1), otherwise scale_ratio_u = new_width / old_width. 
        Remember to calculate scale_ratio_u/v consistently with align_corners before passing into this function. 
        """
        if scale_ratio_u is None and scale_ratio_v is None:
            assert new_u is not None and new_v is not None
            scale_ratio_u = (new_u - 1) / (self.u_size - 1) if align_corners else new_u / self.u_size
            scale_ratio_v = (new_v - 1) / (self.v_size - 1) if align_corners else new_v / self.v_size
        else:
            new_u = scale_ratio_u * (self.u_size - 1) + 1 if align_corners else scale_ratio_u * self.u_size
            new_v = scale_ratio_v * (self.v_size - 1) + 1 if align_corners else scale_ratio_v * self.v_size

        if self.mode == "from_KRt":
            T = self.T.copy()
            K = self.K.copy()
            K[0,0] = K[0,0] * scale_ratio_u
            K[1,1] = K[1,1] * scale_ratio_v
            if align_corners:
                K[0,2] = K[0,2] * scale_ratio_u
                K[1,2] = K[1,2] * scale_ratio_v
            else:
                K[0,2] = (K[0,2] + 0.5) * scale_ratio_u - 0.5
                K[1,2] = (K[1,2] + 0.5) * scale_ratio_v - 0.5
            new_calib = Calib(K=K, T=T, u_size=new_u, v_size=new_v)
        elif self.mode == "from_vps":
            vp1 = self.vp1.copy()
            vp2 = self.vp2.copy()
            pp = self.pp.copy()
            if align_corners:
                vp1[0] = vp1[0] * scale_ratio_u
                vp1[1] = vp1[1] * scale_ratio_v
                vp2[0] = vp2[0] * scale_ratio_u
                vp2[1] = vp2[1] * scale_ratio_v
                pp[0] = pp[0] * scale_ratio_u
                pp[1] = pp[1] * scale_ratio_v
            else:
                vp1[0] = (vp1[0] + 0.5) * scale_ratio_u - 0.5
                vp1[1] = (vp1[1] + 0.5) * scale_ratio_v - 0.5
                vp2[0] = (vp2[0] + 0.5) * scale_ratio_u - 0.5
                vp2[1] = (vp2[1] + 0.5) * scale_ratio_v - 0.5
                pp[0] = (pp[0] + 0.5) * scale_ratio_u - 0.5
                pp[1] = (pp[1] + 0.5) * scale_ratio_v - 0.5
            new_calib = Calib(vp1=vp1, vp2=vp2, height=self.height, u_size=new_u, v_size=new_v, pp=pp)
        else:
            pts_world = self.pts_world.copy()
            pts_image = self.pts_image.copy()
            if align_corners:
                pts_image[:,0] = pts_image[:,0] * scale_ratio_u
                pts_image[:,1] = pts_image[:,1] * scale_ratio_v
            else:
                pts_image[:,0] = (pts_image[:,0] + 0.5) * scale_ratio_u - 0.5
                pts_image[:,1] = (pts_image[:,1] + 0.5) * scale_ratio_v - 0.5
            new_calib = Calib(pts_image=pts_image, pts_world=pts_world, u_size=new_u, v_size=new_v)

        return new_calib

    def pad(self, pad_left, pad_top, pad_right, pad_bottom):
        """Although only pad_left and pad_top affect the intrinsic K, pad_right and pad_bottom are used to update u_size and v_size. 
        pad parameters can be negative. """
        new_u = self.u_size + pad_left + pad_right
        new_v = self.v_size + pad_top + pad_bottom
        if self.mode == "from_KRt":
            T = self.T.copy()
            K = self.K.copy()
            K[0,2] = K[0,2] + pad_left
            K[1,2] = K[1,2] + pad_top
            new_calib = Calib(K=K, T=T, u_size=new_u, v_size=new_v)
        elif self.mode == "from_vps":
            vp1 = self.vp1.copy()
            vp2 = self.vp2.copy()
            pp = self.pp.copy()
            vp1[0] += pad_left
            vp1[1] += pad_top
            vp2[0] += pad_left
            vp2[1] += pad_top
            pp[0] += pad_left
            pp[1] += pad_top
            new_calib = Calib(vp1=vp1, vp2=vp2, height=self.height, u_size=new_u, v_size=new_v, pp=pp)
        else:
            pts_world = self.pts_world.copy()
            pts_image = self.pts_image.copy()
            pts_image[:,0] = pts_image[:,0] + pad_left
            pts_image[:,1] = pts_image[:,1] + pad_top
            new_calib = Calib(pts_image=pts_image, pts_world=pts_world, u_size=new_u, v_size=new_v)

        return new_calib
            
    def flip(self, lr=False, tb=False):
        """Here, the flipping is entirely embedded in K. 
        Be careful later when extracting the fx, fy elements individually for calculation. Here it may not be a problem since we only use H_world_img from this class. 
        In c3d, flip only changes cx, cy, not fx, fy, because the 3D point cloud is also flipped. """
        if self.mode == "from_KRt":
            T = self.T.copy()
            K = self.K.copy()
            if lr:
                K[0,2] = self.u_size - 1 - K[0,2]
                K[0,0] = - K[0,0]
            if tb:
                K[1,2] = self.v_size - 1 - K[1,2]
                K[1,1] = - K[1,1]
            new_calib = Calib(K=K, T=T, u_size=self.u_size, v_size=self.v_size)
        elif self.mode == "from_vps":
            vp1 = self.vp1.copy()
            vp2 = self.vp2.copy()
            pp = self.pp.copy()
            if lr:
                vp1[0] = self.u_size - 1 - vp1[0]
                vp2[0] = self.u_size - 1 - vp2[0]
                pp[0] = self.u_size - 1 - pp[0]
            if tb:
                vp1[1] = self.v_size - 1 - vp1[1]
                vp2[1] = self.v_size - 1 - vp2[1]
                pp[1] = self.v_size - 1 - pp[1]
            new_calib = Calib(vp1=vp1, vp2=vp2, height=self.height, u_size=self.u_size, v_size=self.v_size, pp=pp)
        else:
            pts_world = self.pts_world.copy()
            pts_image = self.pts_image.copy()
            if lr:
                pts_image[:,0] = self.u_size - 1 - pts_image[:,0]
            if tb:
                pts_image[:,1] = self.v_size - 1 - pts_image[:,1]
            new_calib = Calib(pts_image=pts_image, pts_world=pts_world, u_size=self.u_size, v_size=self.v_size)

        return new_calib

