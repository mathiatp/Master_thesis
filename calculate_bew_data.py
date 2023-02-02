import numpy as np
import cv2
from config import MAX_DIST_X, MIN_DIST_X, MAX_DIST_Y, MIN_DIST_Y, BEW_IMAGE_HEIGHT, BEW_IMAGE_WIDTH
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

def calculate_im_pos(height, width, K, camera_rotation, camera_translation, name):
    try:
        file_name_pixel_position = '/home/mathias/Documents/Master_Thesis/pixel_position_arrays/pixel_position_'+name+'.npy'
        im_pos = np.load(file_name_pixel_position)
        return im_pos

    except OSError:
        print('Could not find file: ' + file_name_pixel_position + '. \nStarting to calculate pixel positions for camera ' + name + '.')

        target_elevation = np.array(0)

        xx,yy = np.meshgrid(range(width), range(height), indexing='xy')
        grid = np.array([xx,yy])
        grid = np.einsum('ijk->jki', grid)

        im_pos = np.zeros((height, width,3), dtype=np.float32)

        for row in grid:
            for coor in row:
                pixel = coor
                pos = georeference_point_eq(K, pixel, camera_rotation, camera_translation, target_elevation).astype(np.float32)
                im_pos[int(pixel[1]),int(pixel[0])] = np.concatenate((pos[0:2],np.array([1])))

        np.save(file_name_pixel_position, im_pos)     
        return im_pos

def normalize_im_pos_for_BEW(camera_rotation, im_pos_cut_t):
    im_pos_cut_F = im_pos_cut_t
    # Swithing from x forward, y right axis to x right, y forward to match image cooridnates
    switch_xy = np.array([[0,1,0],
                            [1,0,0],
                            [0,0,1]])

    im_pos_cut_F = np.einsum('ij, jkl->ikl', switch_xy, im_pos_cut_F)

    max_x = MAX_DIST_X
    min_x = MIN_DIST_X
    
    max_y = MAX_DIST_Y
    min_y = MIN_DIST_Y

    s_x = 2/(max_x-min_x)
    t_x = -s_x * min_x-1

    s_y = 2/(max_y-min_y)
    t_y = -s_y * min_y-1

    N = np.array([[s_x, 0,  t_x],
                    [0, s_y,  t_y],
                    [0, 0,    1]])
    im_pos_cut_F_normalized = np.einsum('ij,jkl->ikl', N, im_pos_cut_F)

    return im_pos_cut_F_normalized



def calculate_BEW_points_and_rgb_for_interpolation(camera_rotation, img_pos, img_undistorted):
    row_cut_off = 526
    rgb_chan = 3
    im_pos_cut = img_pos[row_cut_off:,:,:]
    im_pos_cut_T = np.einsum('ijk->kij',im_pos_cut)
    
    im_cut = img_undistorted[row_cut_off:,:,:]
    im_pos_normalized = normalize_im_pos_for_BEW(camera_rotation,im_pos_cut_T)


    new_h, new_w = 1024, 1224

    K = np.array([[new_w/2-1,     0,          new_w/2],
                    [0,           -(new_h/2-1),    new_h/2],
                    [0,           0,          1]])

    im_pos_pixel = np.einsum('ij,jkl->ikl', K, im_pos_normalized)
    im_pos_pixel = np.einsum('ijk->jki',im_pos_pixel)

    im_pos_pixel =  np.nan_to_num(im_pos_pixel, nan = 99999999)
    im_pos_pixel = im_pos_pixel.astype(int)
    im_pos_pixel = im_pos_pixel[:,:,:2]

    points_x_all = im_pos_pixel[:,:,1]
    points_x = np.transpose(np.array([np.ravel(points_x_all)]))
    points_x_all = np.transpose(np.array([np.ravel(points_x_all)]))
    points_x = np.transpose(np.array([points_x[points_x != 99999999]]))

    points_y = im_pos_pixel[:,:,0]
    points_y = np.transpose(np.array([np.ravel(points_y)]))
    points_y = np.transpose(np.array([points_y[points_y != 99999999]]))

    # points_y = np.transpose(np.array([np.ravel(im_pos_pixel[im_pos_pixel[:,:,0] != 99999999])]))
    points = np.concatenate((points_x,points_y), axis=1)
    grid_x,grid_y = np.meshgrid(range(new_h), range(new_w), indexing='ij')

    rgb = im_cut

    rgb = np.reshape(rgb,(len(points_x_all), rgb_chan))
    rgb = np.delete(rgb, np.where(points_x_all == 99999999), axis=0)

    # grid_z0 = griddata(points, rgb, (grid_x, grid_y), method='linear')
    # grid_z0[np.where(np.isnan(grid_z0))] = 0
    # grid_z0 = grid_z0[:,:,:].astype(np.uint8)
    
    return  points, rgb 