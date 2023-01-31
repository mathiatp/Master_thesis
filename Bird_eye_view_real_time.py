from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rosbags.image import message_to_cvimage
from utils import georeference_point_eq
from scipy.interpolate import griddata
import yaml
from yaml.loader import SafeLoader
from cls_mA2 import mA2
from cls_Camera import Camera
from calculate_bew_data import calculate_BEW_points_and_rgb_for_interpolation
from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH

def make_BEW(vessel_mA2: mA2):
    points_fp_f, rgb_fp_f = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.fp_f.camera_rotation, vessel_mA2.fp_f.pixel_positions, vessel_mA2.fp_f.im)
    points_fs_f, rgb_fs_f = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.fs_f.camera_rotation, vessel_mA2.fs_s.pixel_positions, vessel_mA2.fs_f.im)
    points_fs_s, rgb_fs_s = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.fs_s.camera_rotation, vessel_mA2.fs_s.pixel_positions, vessel_mA2.fs_s.im)
    points_ap_p, rgb_ap_p = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.ap_p.camera_rotation, vessel_mA2.ap_p.pixel_positions, vessel_mA2.ap_p.im)
    points_ap_a, rgb_ap_a = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.ap_a.camera_rotation, vessel_mA2.ap_a.pixel_positions, vessel_mA2.ap_a.im)
    points_as_a, rgb_as_a = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.as_a.camera_rotation, vessel_mA2.as_a.pixel_positions, vessel_mA2.as_a.im)
    points_as_s, rgb_as_s = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.as_s.camera_rotation, vessel_mA2.as_s.pixel_positions, vessel_mA2.as_s.im)

    points = np.vstack((points_fp_f, points_fs_f,points_fs_s, points_ap_p, points_ap_a,points_as_a,points_as_s))

    rgb = np.vstack((rgb_fp_f, rgb_fs_f,rgb_fs_s, rgb_ap_p, rgb_ap_a,rgb_as_a,rgb_as_s))

    grid_x,grid_y = np.meshgrid(range(BEW_IMAGE_HEIGHT), range(BEW_IMAGE_WIDTH), indexing='ij')


    grid_z0 = griddata(points, rgb, (grid_x, grid_y), method='linear')
    grid_z0[np.where(np.isnan(grid_z0))] = 0
    grid_z0 = grid_z0[:,:,:].astype(np.uint8)
    return grid_z0



def init_mA2():
    # topic_names = ['/rgb_cam_fp_p/image_raw',
    #                '/rgb_cam_fs_f/image_raw',
    #                '/rgb_cam_fs_s/image_raw',
    #                '/rgb_cam_ap_p/image_raw', 
    #                '/rgb_cam_ap_a/image_raw',
    #                '/rgb_cam_as_a/image_raw',
    #                '/rgb_cam_as_s/image_raw']
                   # Missing '/rgb_cam_fp_p/image_raw' in rosbag2...80

    camera_fp_f = Camera('rgb_cam_fp_f')
    camera_fs_f = Camera('rgb_cam_fs_f')
    camera_fs_s = Camera('rgb_cam_fs_s')
    camera_ap_p = Camera('rgb_cam_ap_p')
    camera_ap_a = Camera('rgb_cam_ap_a')
    camera_as_f = Camera('rgb_cam_as_a')
    camera_as_s = Camera('rgb_cam_as_s')
    vessel_mA2 = mA2(camera_fp_f,camera_fs_f,camera_fs_s,camera_ap_p,camera_ap_a,camera_as_f,camera_as_s)

    return vessel_mA2


def main():
    vessel_mA2 = init_mA2()
    
    file_path = "/home/mathias/Documents/Master_Thesis/Rosbags/rosbag2_2022_10_09-08_02_54_80/"

    topic_name = '/rgb_cam_fp_p/image_raw'

    topic_names = ['/rgb_cam_fp_f/image_raw',
                   '/rgb_cam_fs_f/image_raw',
                   '/rgb_cam_fs_s/image_raw',
                   '/rgb_cam_ap_p/image_raw', 
                   '/rgb_cam_ap_a/image_raw',
                   '/rgb_cam_as_a/image_raw',
                   '/rgb_cam_as_s/image_raw']
    image_count = 0
    
    # create reader instance and open for reading
    with Reader(file_path) as reader:

        connections = [x for x in reader.connections if x.topic in topic_names]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            id = msg.header.frame_id
            print(msg.header.frame_id)
            # TODO check if python 3.10 with match/case works with ROS2
            # msg is rosbags Image or CompressedImage message instance
           
            # get opencv image and convert to bgr8 color space
            img = message_to_cvimage(msg, 'rgb8')
            # plt.figure('Img_distorted')
            # plt.imshow(img)
            # plt.show()
                
            if(id =='rgb_cam_fp_f'):
                vessel_mA2.fp_f._im = cv2.undistort(img,vessel_mA2.fp_f.K,vessel_mA2.fp_f.D)
     
            elif(id =='rgb_cam_fs_f'):
                vessel_mA2.fs_f._im = cv2.undistort(img,vessel_mA2.fs_f.K,vessel_mA2.fs_f.D)

            elif(id =='rgb_cam_fs_s'):
                vessel_mA2.fs_s._im = cv2.undistort(img,vessel_mA2.fs_s.K,vessel_mA2.fs_s.D)

            elif(id =='rgb_cam_ap_p'):
                vessel_mA2.ap_p._im = cv2.undistort(img,vessel_mA2.ap_p.K,vessel_mA2.ap_p.D)
            
            elif(id =='rgb_cam_ap_a'):
                vessel_mA2.ap_a._im = cv2.undistort(img,vessel_mA2.ap_a.K,vessel_mA2.ap_a.D)
            
            elif(id =='rgb_cam_as_a'):
                vessel_mA2.as_a._im = cv2.undistort(img,vessel_mA2.as_a.K,vessel_mA2.as_a.D)
            
            elif(id =='rgb_cam_as_s'):
                vessel_mA2.as_s._im = cv2.undistort(img,vessel_mA2.as_s.K,vessel_mA2.as_s.D)
            image_count = image_count + 1
            
            if image_count >= 12:
                print('Making BEW')
                img_bew = make_BEW(vessel_mA2)

            # plt.figure('Img_distorted')
            # plt.imshow(img)
            # plt.figure('Img_undistorted')
            # plt.imshow(img_undisorted)
                plt.figure('BEW')
                plt.imshow(img_bew)
                plt.show()

            

  
    

if __name__ == '__main__':
    main()


# rgb_cam_ap_a
# rgb_cam_ap_p
# rgb_cam_fs_f
# rgb_cam_as_a
# rgb_cam_fp_f
# rgb_cam_as_s
# rgb_cam_ap_a
# rgb_cam_fs_s
# rgb_cam_ap_p
# rgb_cam_fs_f
# rgb_cam_as_a
# rgb_cam_fp_f
# rgb_cam_as_s
# rgb_cam_ap_a
# rgb_cam_fs_s
# rgb_cam_ap_p
# rgb_cam_fs_f
# rgb_cam_as_a
# rgb_cam_fp_f
# rgb_cam_as_s