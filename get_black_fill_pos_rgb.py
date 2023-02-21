import numpy as np
import matplotlib.pyplot as plt
import cv2

from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH

def get_black_pixel_pos_and_rgb():
    im_red = cv2.imread('/home/mathias/Documents/Master_Thesis/Images/Ros2/BEW_main_func_1_frame_1500x1500px_step_3_8_cameras_mask_red.png')
    h,w,c = np.shape(im_red)
    if((h!=BEW_IMAGE_HEIGHT) and  (w != BEW_IMAGE_WIDTH)):
        im_red = cv2.resize(im_red,(BEW_IMAGE_HEIGHT,BEW_IMAGE_WIDTH))
        
    pixel_pos_black = np.argwhere(cv2.inRange(im_red,(0,0,255),(0,0,255)))
    rgb_black = np.zeros((np.shape(pixel_pos_black)[0],3))
    return pixel_pos_black, rgb_black

def im_mask_walls(name):
    file_name = '/home/mathias/Documents/Master_Thesis/Images/Mask_wall/im_'+name[-4:]+'_mask_wall.png'
    try:
        im_red = cv2.imread(file_name)
    except cv2.error as e:
        print(e)
    if im_red is None:
        print('ERROR: Could not read file: '+ file_name)
        print('Check if file is present in correct folder and under correct name.')
        exit()

    mask_red = np.ones((np.shape(im_red)[0], np.shape(im_red)[1]), dtype=bool)

    mask_red[np.all(im_red == (0,0,255), axis=-1)] = False
    # im_red[np.all(im_red == (0,0,255), axis=-1)] = (255,255,255)
    # pixel_pos_wall = np.argwhere(cv2.inRange(im_red,(0,0,255),(0,0,255)))
    # im[np.all(mask_red==False)] = np.nan
    # rgb_black = np.zeros((np.shape(pixel_pos_black)[0],3))

    return mask_red

# im_mask_walls()