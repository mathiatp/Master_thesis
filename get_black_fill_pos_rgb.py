import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_black_pixel_pos_and_rgb():
    im_red = cv2.imread('/home/mathias/Documents/Master_Thesis/Images/Ros2/First_360_view_7_cameras_red_mask.png')
    pixel_pos_black = np.argwhere(cv2.inRange(im_red,(0,0,255),(0,0,255)))
    rgb_black = np.zeros((np.shape(pixel_pos_black)[0],3))
    return pixel_pos_black, rgb_black
