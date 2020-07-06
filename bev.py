import numpy as np

from .frozen_class import FrozenClass
from .homography import homo_from_pts

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
        assert self.x_size == self.x_max - self.x_min
        assert self.y_size == self.y_max - self.y_min
        assert self.u_axis in ["x", "y", "-x", "-y"]
        assert self.v_axis in ["x", "y", "-x", "-y"]
        assert ("x" in self.u_axis and "y" in self.v_axis ) or ("y" in self.u_axis and "x" in self.v_axis)
        
    def gen_H_world_bev(self):
        self.check_validity()

        pts_img = np.array([[0, 0], [0, self.v_size], [self.u_size, self.v_size], [self.u_size, 0]], dtype=np.float)
        pts_world = self.gen_bev_corners_in_world()

        H_world_bev = homo_from_pts(pts_img, pts_world)

        return H_world_bev

    def gen_bev_corners_in_world(self):

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
            raise ValueError("illegal u_axis and v_axis combo", u_axis, v_axis)
        pts_world = pts_world[world_idx]

        return pts_world


if __name__ == "__main__":
    bev_world_spec = BEVWorldSpec(x_size=10)
    # bev_world_spec.x_size = 10
    print(bev_world_spec.x_size)