import numpy as np
from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH
from file_handling import mA2_frame_yaml_to_rotation_translation_vec, camera_name_calib_yaml_to_K_D
from calculate_bew_data import calculate_im_pos



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
        

    def update_camera_roation_and_translation(self, dict):
        camera_key = 'eo_'+ self.name.lower()

        rotation = np.array(dict[camera_key]['static_transform']['rotation'])
        translation = np.array([dict[camera_key]['static_transform']['translation']])

        self._camera_rotation = rotation
        self._camera_translation = translation


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
