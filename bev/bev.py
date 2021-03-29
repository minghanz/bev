import numpy as np

from .frozen_class import FrozenClass
from .homo import homo_from_pts

class BEVWorldSpec(FrozenClass):
    def __init__(self, u_size, v_size,  **kwargs):
        self.u_size = u_size
        self.v_size = v_size
        self.u_axis = "-x"
        self.v_axis = "y"

        self.x_size = None
        self.y_size = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        ### these 4 are for describing the correspondence between bev and world more precisely after scaling and or padding
        self.u_min = None
        self.u_max = None
        self.v_min = None
        self.v_max = None

        self._freeze()
        self.__dict__.update(kwargs)

        self.update()
        self.check_validity()

    def set_keep(self, **kwargs):
        """when updating a value, need to set to a related one to None at the same time"""
        self.__dict__.update(kwargs)

        self.update()
        
    def update(self):
        """when one is None we update, otherwise all should not be None and should be consistent"""
        x_array = np.array([self.x_min, self.x_max, self.x_size])
        if any(x_array==None):
            assert (x_array==None).sum() == 1, x_array
            self.x_min = self.x_max - self.x_size if self.x_min is None else self.x_min
            self.x_max = self.x_min + self.x_size if self.x_max is None else self.x_max
            self.x_size = self.x_max - self.x_min if self.x_size is None else self.x_size
        else:
            assert self.x_size == self.x_max - self.x_min

        y_array = np.array([self.y_min, self.y_max, self.y_size])
        if any(y_array==None):
            assert (y_array==None).sum() == 1, y_array
            self.y_min = self.y_max - self.y_size if self.y_min is None else self.y_min
            self.y_max = self.y_min + self.y_size if self.y_max is None else self.y_max
            self.y_size = self.y_max - self.y_min if self.y_size is None else self.y_size
        else:
            assert self.y_size == self.y_max - self.y_min
        
    def check_validity(self):
        assert all(np.array([self.x_min, self.x_max, self.x_size])!=None)
        assert all(np.array([self.y_min, self.y_max, self.y_size])!=None)
        assert np.isclose(self.x_size, self.x_max - self.x_min)
        assert np.isclose(self.y_size, self.y_max - self.y_min)
        assert self.u_axis in ["x", "y", "-x", "-y"]
        assert self.v_axis in ["x", "y", "-x", "-y"]
        assert ("x" in self.u_axis and "y" in self.v_axis ) or ("y" in self.u_axis and "x" in self.v_axis)
        
    def gen_H_world_bev(self):
        self.check_validity()

        if self.u_min is None:  # original default behavior
            pts_img = np.array([[0, 0], [0, self.v_size], [self.u_size, self.v_size], [self.u_size, 0]], dtype=np.float)
        else:   ### after scaling or padding the bev, the corresponding points may change (describing the correspondence between bev and world more precisely after scaling and or padding)
            pts_img = np.array([[self.u_min, self.v_min], [self.u_min, self.v_max], [self.u_max, self.v_max], [self.u_max, self.v_min]], dtype=np.float)

        pts_world = self.gen_bev_corners_in_world()

        H_world_bev = homo_from_pts(pts_img, pts_world)

        return H_world_bev

    def gen_bev_corners_in_world(self):
        """top-left, bottom-left, bottom-right, top-right in bev"""
        pts_world = np.array([[self.x_min, self.y_min], [self.x_min, self.y_max], [self.x_max, self.y_max], [self.x_max, self.y_min]], dtype=np.float)

        if self.u_axis == "x" and self.v_axis == "y":
            world_idx = [0,1,2,3]
        elif self.u_axis == "x" and self.v_axis == "-y":
            world_idx = [1,0,3,2]
        elif self.u_axis == "-x" and self.v_axis == "-y":
            world_idx = [2,3,0,1]
        elif self.u_axis == "-x" and self.v_axis == "y":
            world_idx = [3,2,1,0]
        elif self.u_axis == "y" and self.v_axis == "x":
            world_idx = [0,3,2,1]
        elif self.u_axis == "y" and self.v_axis == "-x":
            world_idx = [3,0,1,2]
        elif self.u_axis == "-y" and self.v_axis == "-x":
            world_idx = [2,1,0,3]
        elif self.u_axis == "-y" and self.v_axis == "x":
            world_idx = [1,2,3,0]
        else:
            raise ValueError("illegal u_axis and v_axis combo", self.u_axis, self.v_axis)
        pts_world = pts_world[world_idx]

        return pts_world

    def scale(self, align_corners, new_u=None, new_v=None, scale_ratio_u=None, scale_ratio_v=None):
        """This is to generate a new BEVWorldSpec object based on self object, according to scaling parameters. 
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

        u_min = self.u_min if self.u_min is not None else 0
        u_max = self.u_max if self.u_max is not None else self.u_size
        v_min = self.v_min if self.v_min is not None else 0
        v_max = self.v_max if self.v_max is not None else self.v_size

        if align_corners:
            u_min = u_min * scale_ratio_u
            u_max = u_max * scale_ratio_u
            v_min = v_min * scale_ratio_v
            v_max = v_max * scale_ratio_v
        else:
            u_min = (u_min + 0.5) * scale_ratio_u - 0.5
            u_max = (u_max + 0.5) * scale_ratio_u - 0.5
            v_min = (v_min + 0.5) * scale_ratio_v - 0.5
            v_max = (v_max + 0.5) * scale_ratio_v - 0.5

        new_bspec = BEVWorldSpec(u_size=new_u, v_size=new_v, u_axis=self.u_axis, v_axis=self.v_axis, x_size=self.x_size, y_size=self.y_size, x_min=self.x_min, y_min=self.y_min, 
                    u_min=u_min, v_min=v_min, u_max=u_max, v_max=v_max)

        return new_bspec        

    def pad(self, pad_left, pad_top, pad_right, pad_bottom):
        
        new_u = self.u_size + pad_left + pad_right
        new_v = self.v_size + pad_top + pad_bottom

        u_min = self.u_min if self.u_min is not None else 0
        u_max = self.u_max if self.u_max is not None else self.u_size
        v_min = self.v_min if self.v_min is not None else 0
        v_max = self.v_max if self.v_max is not None else self.v_size

        u_min = u_min + pad_left
        u_max = u_max + pad_left
        v_min = v_min + pad_top
        v_max = v_max + pad_top

        new_bspec = BEVWorldSpec(u_size=new_u, v_size=new_v, u_axis=self.u_axis, v_axis=self.v_axis, x_size=self.x_size, y_size=self.y_size, x_min=self.x_min, y_min=self.y_min, 
                    u_min=u_min, v_min=v_min, u_max=u_max, v_max=v_max)

        return new_bspec

    def flip(self, lr=False, tb=False):
        u_axis = self.u_axis
        v_axis = self.v_axis
        
        if lr:
            u_axis = '-' + u_axis if '-' not in u_axis else u_axis[1]
        if tb:
            v_axis = '-' + v_axis if '-' not in v_axis else v_axis[1]

        new_bspec = BEVWorldSpec(u_size=self.u_size, v_size=self.v_size, x_size=self.x_size, y_size=self.y_size, x_min=self.x_min, y_min=self.y_min, 
                    u_min=self.u_min, v_min=self.v_min, u_max=self.u_max, v_max=self.v_max, 
                    u_axis=u_axis, v_axis=v_axis)
        
        return new_bspec

if __name__ == "__main__":
    bev_world_spec = BEVWorldSpec(x_size=10)
    # bev_world_spec.x_size = 10
    print(bev_world_spec.x_size)