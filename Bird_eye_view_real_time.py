from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rosbags.image import message_to_cvimage
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay
import yaml
import scipy
from yaml.loader import SafeLoader
from cls_mA2 import mA2
from cls_Camera import Camera
from calculate_bew_data import calculate_BEW_points_and_rgb_for_interpolation, interpolate, interp_weights
from get_black_fill_pos_rgb import get_black_pixel_pos_and_rgb
from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH
import cProfile
import pstats
import multiprocessing
import time

#The main advantage of Python is that there are a number of ways of very easily 
# extending your code with C (ctypes, swig,f2py) / C++ (boost.python, weave.inline, weave.blitz) / Fortran (f2py) 
# - or even just by adding type annotations to Python so it can be processed to C (cython)
def individual_color_interpolation(grid_x: np.array,
                                   grid_y: np.array,
                                   delaunay: Delaunay,
                                   color: np.array):
    interp = NearestNDInterpolator(delaunay,color)
    im_color = interp((grid_x, grid_y))
    return im_color

def individual_color_interpolation_mp(args):
    grid_x, grid_y, delaunay, color, i = args
    interp = NearestNDInterpolator(delaunay,color)
    im_color = interp((grid_x, grid_y))
    print('Color: '+ str(i))
    return [im_color,i]

def make_BEW(vessel_mA2: mA2):
    start = time.time()
    points_fp_f, rgb_fp_f = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.fp_f.camera_rotation, vessel_mA2.fp_f.pixel_positions, vessel_mA2.fp_f.im)
    points_fs_f, rgb_fs_f = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.fs_f.camera_rotation, vessel_mA2.fs_f.pixel_positions, vessel_mA2.fs_f.im)
    points_fs_s, rgb_fs_s = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.fs_s.camera_rotation, vessel_mA2.fs_s.pixel_positions, vessel_mA2.fs_s.im)
    points_ap_p, rgb_ap_p = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.ap_p.camera_rotation, vessel_mA2.ap_p.pixel_positions, vessel_mA2.ap_p.im)
    points_ap_a, rgb_ap_a = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.ap_a.camera_rotation, vessel_mA2.ap_a.pixel_positions, vessel_mA2.ap_a.im)
    points_as_a, rgb_as_a = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.as_a.camera_rotation, vessel_mA2.as_a.pixel_positions, vessel_mA2.as_a.im)
    points_as_s, rgb_as_s = calculate_BEW_points_and_rgb_for_interpolation(vessel_mA2.as_s.camera_rotation, vessel_mA2.as_s.pixel_positions, vessel_mA2.as_s.im)
    
    grid_x,grid_y = np.meshgrid(range(BEW_IMAGE_HEIGHT), range(BEW_IMAGE_WIDTH), indexing='ij')

    points = np.vstack((points_fp_f,
                        points_fs_f,
                        points_fs_s, 
                        points_ap_p, 
                        points_ap_a,
                        points_as_a,
                        points_as_s,
                        vessel_mA2.black_pixel_pos))
    # np.save('points.npy',points)

    rgb = np.vstack((rgb_fp_f,
                     rgb_fs_f, 
                     rgb_fs_s, 
                     rgb_ap_p, 
                     rgb_ap_a,
                     rgb_as_a,
                     rgb_as_s,
                     vessel_mA2.black_pixel_rgb))

    # np.save('rgb.npy',rgb)
    # # Delaunay 1
    # interp = NearestNDInterpolator(vessel_mA2.delaunay,rgb)
    # interp.
    # im = interp((grid_x, grid_y))

    
    # input_for_mp = [[grid_x,grid_y,vessel_mA2.delaunay,rgb[:,0],0],
    #                 [grid_x,grid_y,vessel_mA2.delaunay,rgb[:,1],1],
    #                 [grid_x,grid_y,vessel_mA2.delaunay,rgb[:,2],2]]
    
    # Delaunay 2
    # red = individual_color_interpolation(grid_x,grid_y,vessel_mA2.delaunay,rgb[:,0])
    # green = individual_color_interpolation(grid_x,grid_y,vessel_mA2.delaunay,rgb[:,1])
    # blue = individual_color_interpolation(grid_x,grid_y,vessel_mA2.delaunay,rgb[:,2])
    # im = np.dstack((red,green,blue))
    
    # Delaunay 3
    # with multiprocessing.Pool() as pool:
    #     color = pool.map(individual_color_interpolation_mp,input_for_mp)
    # im = np.dstack((color[0][0],color[1][0],color[2][0]))
    # end = time.time()
    # print('Time: ' + str(end-start))
    # return im

    # Delaunay 4
    # xy = points
    # uv=np.zeros([grid_x.shape[0]*grid_y.shape[1],2])
    # uv[:,0]=grid_y.flatten()
    # uv[:,1]=grid_x.flatten()
    values = rgb

    # Computed once and for all !
    # vtx, wts = interp_weights(xy, uv)
    # np.save('vtx.npy', vtx)
    # np.save('wts.npy', wts)

    vtx = np.load('vtx.npy')
    wts = np.load('wts.npy')
    start = time.time()
    valuesi_r=interpolate(values[:,0].flatten(), vtx, wts)
    valuesi_g=interpolate(values[:,1].flatten(), vtx, wts)
    valuesi_b=interpolate(values[:,2].flatten(), vtx, wts)
    # end = time.time()
    # print('Time: ' + str(end-start))

    valuesi_r = valuesi_r.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_r = valuesi_r.astype(np.uint8)
    
    valuesi_g= valuesi_g.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_g = valuesi_g.astype(np.uint8)
    
    

    valuesi_b= valuesi_b.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_b = valuesi_b.astype(np.uint8)
 
    im = np.dstack((valuesi_r,valuesi_g,valuesi_b))
    end = time.time()
    print('Time: ' + str(end-start))
    return im


    # Griddata
    # grid_z0 = griddata(points, rgb[:], (grid_x, grid_y), method='nearest')
    # grid_z0[np.where(np.isnan(grid_z0))] = 0
    # grid_z0 = grid_z0[:,:,:].astype(np.uint8)
    # end = time.time()
    # print('Time: ' + str(end-start))
    # return grid_z0



def init_mA2():
    # print(scipy.__version__) 1.9.1
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
    # print('Cores:'+str(multiprocessing.cpu_count())) = 12
    
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
    frame = 0
    
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
            
            if image_count %8 == 0:
                print('Making BEW')
                img_bew = make_BEW(vessel_mA2)
                frame = frame + 1

            # plt.figure('Img_distorted')
            # plt.imshow(img)
                # plt.imsave('First 360 view_7_cameras.png', img_bew)
                # plt.figure('Img_undistorted_as_a')
                # plt.imshow(vessel_mA2.as_a.im)
                # plt.figure('BEW')
                # plt.imshow(img_bew)
                # plt.show()
            if frame >=10:
                break
            


            

  
    

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    # stats.dump_stats(filename='./Profiler_stats/Ros2/main_func_10_frame_1500x1500px_nearest_pre_calc_we.prof')
