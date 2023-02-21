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
from calculate_bew_data import calculate_BEW_points_and_rgb_for_interpolation,calculate_rgb_matrix_for_BEW, interp_weights, interpolate, make_final_mask_and_pixel_pos
from config import BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH
import cProfile
import pstats
# import multiprocessing
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

    grid_x,grid_y = np.meshgrid(range(BEW_IMAGE_HEIGHT), range(BEW_IMAGE_WIDTH), indexing='ij')

    points = np.vstack((vessel_mA2.fp_p.pixel_positions_masked,
                        vessel_mA2.fp_f.pixel_positions_masked,
                        vessel_mA2.fs_f.pixel_positions_masked,
                        vessel_mA2.fs_s.pixel_positions_masked,
                        vessel_mA2.ap_p.pixel_positions_masked,
                        vessel_mA2.ap_a.pixel_positions_masked,
                        vessel_mA2.as_a.pixel_positions_masked,
                        vessel_mA2.as_s.pixel_positions_masked,
                        vessel_mA2.black_pixel_pos))

    rgb_fp_p = calculate_rgb_matrix_for_BEW(vessel_mA2.fp_p.im,vessel_mA2.fp_p.image_mask)
    rgb_fp_f = calculate_rgb_matrix_for_BEW(vessel_mA2.fp_f.im,vessel_mA2.fp_f.image_mask)
    rgb_fs_f = calculate_rgb_matrix_for_BEW(vessel_mA2.fs_f.im,vessel_mA2.fs_f.image_mask) 
    rgb_fs_s = calculate_rgb_matrix_for_BEW(vessel_mA2.fs_s.im,vessel_mA2.fs_s.image_mask) 
    rgb_ap_p = calculate_rgb_matrix_for_BEW(vessel_mA2.ap_p.im,vessel_mA2.ap_p.image_mask)
    rgb_ap_a = calculate_rgb_matrix_for_BEW(vessel_mA2.ap_a.im,vessel_mA2.ap_a.image_mask)
    rgb_as_a = calculate_rgb_matrix_for_BEW(vessel_mA2.as_a.im,vessel_mA2.as_a.image_mask)
    rgb_as_s = calculate_rgb_matrix_for_BEW(vessel_mA2.as_s.im,vessel_mA2.as_s.image_mask)

    rgb = np.vstack((rgb_fp_p,
                     rgb_fp_f,
                     rgb_fs_f, 
                     rgb_fs_s, 
                     rgb_ap_p, 
                     rgb_ap_a,
                     rgb_as_a,
                     rgb_as_s,
                     vessel_mA2.black_pixel_rgb))

    # Delaunay 4
    
    xy = points
    values = rgb
    
    uv=np.zeros([grid_x.shape[0]*grid_y.shape[1],2])
    uv1=np.zeros([grid_x.shape[0]*grid_y.shape[1],2])
    st = time.time()
    uv1[:,0]=grid_y.flatten()
    uv1[:,1]=grid_x.flatten()

    val_r_f = values[:,0].flatten()
    val_g_f = values[:,1].flatten()
    val_b_f = values[:,2].flatten()
    et =time.time()
    time_flatten = et-st


    st =time.time()
    uv[:,0]=grid_y.ravel()
    uv[:,1]=grid_x.ravel()
    
    val_r = values[:,0].ravel()
    val_g = values[:,1].ravel()
    val_b = values[:,2].ravel()
    et = time.time()

    time_ravel = et-st
    

    
    if ((vessel_mA2.vtx is None) or (vessel_mA2.wts is None)):
    # Computed once and for all !
        vtx, wts = interp_weights(xy, uv)
        np.save('vtx.npy', vtx)
        np.save('wts.npy', wts)
        
    vtx = vessel_mA2.vtx
    wts = vessel_mA2.wts
    
    
    
    
    valuesi_r=interpolate(val_r, vtx, wts)
    valuesi_g=interpolate(val_g, vtx, wts)
    valuesi_b=interpolate(val_b, vtx, wts)
    
    

    valuesi_r = valuesi_r.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_r = valuesi_r.astype(np.uint8)
    
    valuesi_g= valuesi_g.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_g = valuesi_g.astype(np.uint8)
    
    valuesi_b= valuesi_b.reshape((grid_x.shape[0],grid_x.shape[1]),order='F')
    valuesi_b = valuesi_b.astype(np.uint8)
 
    im = np.dstack((valuesi_r,valuesi_g,valuesi_b))
    end = time.time()
    print('Time: ' + str((end-start)*1000) + ' ms')
    return im , np.array([time_flatten, time_ravel])



def init_mA2():
    # print('Cores:'+str(multiprocessing.cpu_count())) = 12
    # print(scipy.__version__) 1.9.1
    # topic_names = ['/rgb_cam_fp_p/image_raw',
    #                '/rgb_cam_fs_f/image_raw',
    #                '/rgb_cam_fs_s/image_raw',
    #                '/rgb_cam_ap_p/image_raw', 
    #                '/rgb_cam_ap_a/image_raw',
    #                '/rgb_cam_as_a/image_raw',
    #                '/rgb_cam_as_s/image_raw']
                   # Missing '/rgb_cam_fp_p/image_raw' in rosbag2...80
    camera_fp_p = Camera('rgb_cam_fp_p')
    camera_fp_f = Camera('rgb_cam_fp_f')
    camera_fs_f = Camera('rgb_cam_fs_f')
    camera_fs_s = Camera('rgb_cam_fs_s')
    camera_ap_p = Camera('rgb_cam_ap_p')
    camera_ap_a = Camera('rgb_cam_ap_a')
    camera_as_f = Camera('rgb_cam_as_a')
    camera_as_s = Camera('rgb_cam_as_s')
    vessel_mA2 = mA2(camera_fp_p,camera_fp_f,camera_fs_f,camera_fs_s,camera_ap_p,camera_ap_a,camera_as_f,camera_as_s)

    return vessel_mA2


def main():
    vessel_mA2 = init_mA2()
    vessel_mA2.find_triangle_between_each_cameras()

    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.fp_p.wall_dist_mask, vessel_mA2.fp_p.pixel_positions_I_BEW, vessel_mA2.fp_p.left_triangle, vessel_mA2.fp_p.right_triangle)
    vessel_mA2.fp_p.set_image_mask(image_mask)
    vessel_mA2.fp_p.set_pixel_positions_masked(pixel_pos_masked)

    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.fp_f.wall_dist_mask, vessel_mA2.fp_f.pixel_positions_I_BEW, vessel_mA2.fp_f.left_triangle, vessel_mA2.fp_f.right_triangle)
    vessel_mA2.fp_f.set_image_mask(image_mask)
    vessel_mA2.fp_f.set_pixel_positions_masked(pixel_pos_masked)
    
    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.fs_f.wall_dist_mask, vessel_mA2.fs_f.pixel_positions_I_BEW, vessel_mA2.fs_f.left_triangle, vessel_mA2.fs_f.right_triangle)
    vessel_mA2.fs_f.set_image_mask(image_mask)
    vessel_mA2.fs_f.set_pixel_positions_masked(pixel_pos_masked)
    
    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.fs_s.wall_dist_mask, vessel_mA2.fs_s.pixel_positions_I_BEW, vessel_mA2.fs_s.left_triangle,vessel_mA2.fs_s.right_triangle)
    vessel_mA2.fs_s.set_image_mask(image_mask)
    vessel_mA2.fs_s.set_pixel_positions_masked(pixel_pos_masked)
    
    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.ap_p.wall_dist_mask, vessel_mA2.ap_p.pixel_positions_I_BEW, vessel_mA2.ap_p.left_triangle, vessel_mA2.ap_p.right_triangle)
    vessel_mA2.ap_p.set_image_mask(image_mask)
    vessel_mA2.ap_p.set_pixel_positions_masked(pixel_pos_masked)
    
    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.ap_a.wall_dist_mask, vessel_mA2.ap_a.pixel_positions_I_BEW, vessel_mA2.ap_a.left_triangle,vessel_mA2.ap_a.right_triangle)
    vessel_mA2.ap_a.set_image_mask(image_mask)
    vessel_mA2.ap_a.set_pixel_positions_masked(pixel_pos_masked)    

    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.as_a.wall_dist_mask, vessel_mA2.as_a.pixel_positions_I_BEW, vessel_mA2.as_a.left_triangle, vessel_mA2.as_a.right_triangle)
    vessel_mA2.as_a.set_image_mask(image_mask)
    vessel_mA2.as_a.set_pixel_positions_masked(pixel_pos_masked)  

    image_mask, pixel_pos_masked =make_final_mask_and_pixel_pos(vessel_mA2.as_s.wall_dist_mask, vessel_mA2.as_s.pixel_positions_I_BEW, vessel_mA2.as_s.left_triangle, vessel_mA2.as_s.right_triangle)
    vessel_mA2.as_s.set_image_mask(image_mask)
    vessel_mA2.as_s.set_pixel_positions_masked(pixel_pos_masked)  

    
    file_path = "/home/mathias/Documents/Master_Thesis/Rosbags/rosbag2_2022_10_06-13_50_07_0"

    topic_names = ['/rgb_cam_fp_p/image_raw',
                   '/rgb_cam_fp_f/image_raw',
                   '/rgb_cam_fs_f/image_raw',
                   '/rgb_cam_fs_s/image_raw',
                   '/rgb_cam_ap_p/image_raw', 
                   '/rgb_cam_ap_a/image_raw',
                   '/rgb_cam_as_a/image_raw',
                   '/rgb_cam_as_s/image_raw']
    image_count = 0
    frame = 0
    # video_file_path = "./Video/BEW_restrcture_step3_1st.mp4"
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # writer = cv2.VideoWriter(video_file_path, fourcc, 5, (BEW_IMAGE_WIDTH, BEW_IMAGE_HEIGHT))

    time_flatten_ravel = np.array([[0,0]])
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

            if(id =='rgb_cam_fp_p'):
                vessel_mA2.fp_p._im = cv2.undistort(img,vessel_mA2.fp_p.K,vessel_mA2.fp_p.D)

            elif(id =='rgb_cam_fp_f'):
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
            
            if (image_count %8 == 0 and image_count > 20): # 11500 for image for thesis
                print('Making BEW')
                img_bew, time_arr = make_BEW(vessel_mA2)
                # writer.write(cv2.cvtColor(img_bew, cv2.COLOR_BGR2RGB))
                frame = frame + 1
                time_flatten_ravel = np.vstack((time_flatten_ravel,time_arr))
                

            # plt.figure('Img_distorted')
            # plt.imshow(img)
                # plt.imsave('Images/Ros2/BEW_main_func_1_frame_1500x1500px_step_3_restructure_all_8_cameras.png', img_bew)
                # plt.imsave('BEW_step_1_restructure_mask.png', img_bew)
                # plt.figure('Img_undistorted_as_a')
                # plt.imshow(vessel_mA2.as_a.im)
                # plt.figure('BEW')
                # plt.imshow(img_bew)
                # plt.show()
            if (frame >=500):
                break

    time_flatten_ravel = np.delete(time_flatten_ravel, 0,0)
    np.save('time_flatten_ravel.npy',time_flatten_ravel)
    # writer.release()         
            


            

  
    

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    # stats.dump_stats(filename='./Profiler_stats/Ros2/main_func_100_frame_1500x1500px_step_3_3_restructur.prof')
