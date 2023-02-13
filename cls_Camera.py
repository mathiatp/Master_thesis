import numpy as np
from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH
from file_handling import mA2_frame_yaml_to_rotation_translation_vec, camera_name_calib_yaml_to_K_D
from calculate_bew_data import calculate_BEW_points_and_mask, calculate_im_pos
from get_black_fill_pos_rgb import im_mask_walls



def get_camera_name_from_topic_str(topic_str: str):
    """Returns the camera name from the topic string"""
    indices = [i for i, c in enumerate(topic_str) if c == '/']

    start_index = indices[2]+1
    end_index = indices[3]
    name = topic_str[start_index:end_index]
    return name

class Camera:
    def __init__(self, name: str):
        self._name = name
        self._topic = '/'+ name + '/image_raw'
        self._K, self._D = camera_name_calib_yaml_to_K_D(name)
        self._camera_rotation, self._camera_translation = mA2_frame_yaml_to_rotation_translation_vec(name)
        self._im = None
        self._pixel_positions = calculate_im_pos(1024, 
                                           1224, 
                                           self.K, 
                                           self.camera_rotation, 
                                           self.camera_translation, 
                                           self.name)
        self._wall_mask = im_mask_walls(name)
        self._image_mask, self._pixel_positions_masked = calculate_BEW_points_and_mask(self.pixel_positions, self.wall_mask)
        

    @property
    def name(self):
        return self._name
    @property
    def D(self):
        return self._D
    @property
    def K(self):
        return self._K
    @property
    def topic(self):
        return self._topic
    @property
    def im(self):
        return self._im
    @property
    def camera_rotation(self):
        return self._camera_rotation
    @property
    def camera_translation(self):
        return self._camera_translation
    @property
    def pixel_positions(self):
        return self._pixel_positions    
    @property
    def wall_mask(self):
        return self._wall_mask
    @property
    def image_mask(self):
        return self._image_mask
    @property
    def pixel_positions_masked(self):
        return self._pixel_positions_masked 

