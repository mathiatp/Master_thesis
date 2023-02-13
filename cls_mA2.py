from calculate_bew_data import calculate_BEW_points_and_mask, calculate_BEW_points_and_rgb_for_interpolation
from cls_Camera import Camera
from get_black_fill_pos_rgb import get_black_pixel_pos_and_rgb
import numpy as np
from scipy.spatial import Delaunay
import cv2


class mA2:    

    def __init__(self,
                 camera_fp_f: Camera,
                 camera_fs_f: Camera,
                 camera_fs_s: Camera,
                 camera_ap_p: Camera,
                 camera_ap_a: Camera,
                 camera_as_f: Camera,
                 camera_as_s: Camera):
        self._fp_f = camera_fp_f
        self._fs_f = camera_fs_f
        self._fs_s = camera_fs_s
        self._ap_p = camera_ap_p
        self._ap_a = camera_ap_a
        self._as_a = camera_as_f
        self._as_s = camera_as_s
        self._black_pixel_pos, self._black_pixel_rgb = get_black_pixel_pos_and_rgb()
        # self._delaunay = self.init_delaunay()

    def init_delaunay(self):
        # points_fp_f, _ = calculate_BEW_points_and_mask(self.fp_f.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_fp_f.png'))
        # points_fs_f, _ = calculate_BEW_points_and_mask(self.fs_f.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_fs_f.png'))
        # points_fs_s, _ = calculate_BEW_points_and_mask(self.fs_s.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_fs_s.png'))
        # points_ap_p, _ = calculate_BEW_points_and_mask(self.ap_p.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_ap_p.png'))
        # points_ap_a, _ = calculate_BEW_points_and_mask(self.ap_a.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_ap_a.png'))
        # points_as_a, _ = calculate_BEW_points_and_mask(self.as_a.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_as_a.png'))
        # points_as_s, _ = calculate_BEW_points_and_mask(self.as_s.wall_mask, cv2.imread('Images/Delaunay_init_images/delaunay_init_im_as_s.png'))
        points = np.vstack((self.fp_f.pixel_positions_masked,
                            self.fs_f.pixel_positions_masked,
                            self.fs_s.pixel_positions_masked,
                            self.ap_p.pixel_positions_masked,
                            self.ap_a.pixel_positions_masked,
                            self.as_a.pixel_positions_masked,
                            self.as_s.pixel_positions_masked,
                            self.black_pixel_pos))
        
        return Delaunay(points)

    @property
    def fp_f(self):
        return self._fp_f
    @property
    def fs_f(self):
        return self._fs_f
    @property
    def fs_s(self):
        return self._fs_s
    @property
    def ap_p(self):
        return self._ap_p
    @property
    def ap_a(self):
        return self._ap_a
    @property
    def as_a(self):
        return self._as_a
    @property
    def as_s(self):
        return self._as_s
    @property
    def delaunay(self):
        return self._delaunay
    @property
    def black_pixel_pos(self):
        return self._black_pixel_pos
    @property
    def black_pixel_rgb(self):
        return self._black_pixel_rgb