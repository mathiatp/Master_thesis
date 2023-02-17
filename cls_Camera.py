import numpy as np
from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH, ROW_CUTOFF
from file_handling import mA2_frame_yaml_to_rotation_translation_vec, camera_name_calib_yaml_to_K_D
from calculate_bew_data import calculate_BEW_points_and_mask, calculate_im_pos
from get_black_fill_pos_rgb import im_mask_walls
import matplotlib.pyplot as plt



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
        self._wall_dist_mask, self._pixel_positions_I_BEW = calculate_BEW_points_and_mask(self.pixel_positions, self.wall_mask)
        self._corners = self.find_corner_points(self.pixel_positions_I_BEW) #corners = top_left,bottom_left, bottom_right, top_right
        self._left_triangle, self._right_triangle = None, None
        # self._image_mask, self._pixel_positions_masked = calculate_BEW_points_and_mask(self.pixel_positions, self.wall_mask)
        self._image_mask, self._pixel_positions_masked = None, None

    def set_left_triangle(self,corners: np.array):
        self._left_triangle = corners
    def set_right_triangle(self,corners: np.array):
        self._right_triangle = corners

    def set_image_mask(self,image_mask: np.array):
        self._image_mask = image_mask    
    def set_pixel_positions_masked(self,pixel_positions_masked: np.array):
        self._pixel_positions_masked = pixel_positions_masked

    def find_corner_points(self, pixel_pos: np.array):
        if(self.name[8] == 'f'):
            furthest_left = np.array([pixel_pos[1,0,0]-1,pixel_pos[0,0,0]]).transpose()

            closest_left = np.array([pixel_pos[1,-1,0],pixel_pos[0,-1,0]]).transpose()

            closest_right = np.array([pixel_pos[1,-1,-1],pixel_pos[0,-1,-1]]).transpose()

            furthest_right = np.array([pixel_pos[1,0,-1]+1,pixel_pos[0,0,-1]]).transpose()

            
        elif(self.name[8]== 'a'):
            # Neede so x-value is correct from left to right. The same as forwards facing cameras
            furthest_left = np.array([pixel_pos[1,0,0],pixel_pos[0,0,-1]-50]).transpose()

            closest_left = np.array([pixel_pos[1,-1,0],pixel_pos[0,-1,-1]]).transpose()

            closest_right = np.array([pixel_pos[1,-1,-1],pixel_pos[0,-1,0]]).transpose()

            furthest_right = np.array([pixel_pos[1,0,-1],pixel_pos[0,0,0]+50]).transpose()

            

            
        
        corners = np.array([furthest_left,closest_left,closest_right, furthest_right])
        return corners
        # pixel_pos_corners = np.array([[[][]],
        #                               [[][]]])
        # return pixel_pos_corners
        # pixel_pos = self.pixel_positions[ROW_CUTOFF:,:,:2]
        # pixel_pos = np.einsum('ijk->kij',pixel_pos) # 2,498,1224
        


        # # # pixel_pos[np.where(pixel_pos[:,:,:2]>100)] = 100
        # # # pixel_pos[np.where(pixel_pos[:,:,:2]<-100)] = -100
        # # min = np.min(pixel_pos[:,:,1])
        # # # pixel_pos[:,:,1] = pixel_pos[:,:,1]+np.abs(min)
        # # max = np.max(pixel_pos[:,:,1])
        # # # pixel_pos[:,:,1] = pixel_pos[:,:,1]/max*255
        # # plt.figure(self.name)
        # # plt.imshow(pixel_pos[:,:,1])
        

        # if(self.name == 'rgb_cam_fp_f'):
        #     top = np.max(pixel_pos[0,:,:])
        #     top_id = np.where(pixel_pos[0,:,:] == np.max(pixel_pos[0,:,:]))
        #     top_port_id = np.where(pixel_pos[1,top_id[0],:] == np.min(pixel_pos[1,top_id[0],:]))
        #     top_starboard_id = np.where(pixel_pos[1,top_id[0],:] == np.max(pixel_pos[1,top_id[0],:]))

            
        #     bot = np.min(pixel_pos[0,:,:])
        #     bot_id = np.where(pixel_pos[0,:,:] == np.min(pixel_pos[0,:,:]))
        #     bot_port_id = np.where(pixel_pos[1,bot_id[0],:] == np.min(pixel_pos[1,bot_id[0],:]))
        #     bot_starboard_id = np.where(pixel_pos[1,bot_id[0],:] == np.max(pixel_pos[1,bot_id[0],:]))

        # if(self.name == 'rgb_cam_fs_s'):
            
        #     top = np.max(pixel_pos[0,:,:])
        #     top_id = np.where(pixel_pos[0,:,:] == np.max(pixel_pos[0,:,:]))
        #     top_port_id = np.where(pixel_pos[1,top_id[0],:] == np.min(pixel_pos[1,top_id[0],:]))
        #     top_starboard_id = np.where(pixel_pos[1,top_id[0],:] == np.max(pixel_pos[1,top_id[0],:]))

            
        #     bot = np.min(pixel_pos[0,:,:])
        #     bot_id = np.where(pixel_pos[0,:,:] == np.min(pixel_pos[0,:,:]))
        #     bot_port_id = np.where(pixel_pos[1,bot_id[0],:] == np.min(pixel_pos[1,bot_id[0],:]))
        #     bot_starboard_id = np.where(pixel_pos[1,bot_id[0],:] == np.max(pixel_pos[1,bot_id[0],:]))
        #     pass
        # else:
        #     print('ERROR: camera name is incorrect')



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
    def pixel_positions_I_BEW(self):
        return self._pixel_positions_I_BEW 
    @property
    def wall_dist_mask(self):
        return self._wall_dist_mask
    @property
    def corners(self):
        return self._corners
    @property
    def left_triangle(self):
        return self._left_triangle
    @property
    def right_triangle(self):
        return self._right_triangle   
    @property
    def pixel_positions_masked(self):
        return self._pixel_positions_masked 
    @property
    def image_mask(self):
        return self._image_mask 

    