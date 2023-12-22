import cv2 as cv
import numpy as np
import pandas as pd
import random
import os
from scipy.spatial.transform import Rotation


def gen_chessboard_points(chessboard_pattern, cellsize=0.025, dtype=np.float32):
    '''Generate 3D points with regular interval on a plane (a.k.a. chessboard).'''
    obj_pts = cellsize * np.array([(c, r, 0) for r in range(chessboard_pattern[1]) for c in range(chessboard_pattern[0])], dtype=dtype)
    row, col = obj_pts.shape
    obj_pts = obj_pts.reshape(row, 1, col)
    return obj_pts

def random_principal_point(img_size):
    cx = random.randint(img_size[1] / 2 - 10, img_size[1] / 2 + 10)
    cy = random.randint(img_size[0] / 2 - 10, img_size[0] / 2 + 10)

    return cx, cy  

def generate_intrinsic_maxtrix(intrinsic_type, img_size):
    if intrinsic_type == 'P0':
        fx = fy = random.randint(800, 1000)
        cx = img_size[1] / 2
        cy = img_size[0] / 2
    elif intrinsic_type == 'P1':
        fx =random.randint(800, 1000)
        fy = random.randint(800, 1000)
        cx = img_size[1] / 2
        cy = img_size[0] / 2
    elif intrinsic_type == 'P2':
        fx = fy = random.randint(800, 1000)
        cx, cy = random_principal_point(img_size)
    else:
        fx = random.randint(800, 1000)
        fy = random.randint(800, 1000)
        cx, cy = random_principal_point(img_size)
    
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

def generate_dist_coef(dist_type):
    if dist_type.startswith('BC'):
        tmp = np.array([-0.2, 0.1, 0.06, 0.04, 0.08])
        dist = np.zeros(5)
        if dist_type == 'BC1':
            dist[0] = tmp[0]
        elif dist_type == 'BC2':
            dist[0] = tmp[0]
            dist[1] = tmp[1]
        elif dist_type == 'BC3':
            dist[0] = tmp[0]
            dist[1] = tmp[1]
            dist[4] = tmp[4]
        elif dist_type == 'BC4':
            dist = tmp
    else:
        tmp = np.array([-0.2, 0.1, 0.08, 0.06])
        dist = np.zeros(4)
        if dist_type == 'KB1':
            dist[0] = tmp[0]
        elif dist_type == 'KB2':
            dist[0] = tmp[0]
            dist[1] = tmp[1]
        elif dist_type == 'KB3':
            dist[0] = tmp[0]
            dist[1] = tmp[1]
            dist[2] = tmp[2]
        elif dist_type == 'KB4':
            dist = tmp
    return dist

def conv_pose2Rt(ori_xyz, pos_xyz, degrees=True):
    '''Convert the given camera pose to the rotation vector (rvec) and translation vector (tvec).'''
    rvec = -Rotation.from_euler('zyx', ori_xyz[::-1], degrees=degrees).as_rotvec() # '-' is transpose for 'rvec'.
    tvec = -Rotation.from_rotvec(rvec).apply(pos_xyz)
    return rvec.ravel(), tvec.ravel()

def save_img_pts(save_path, img_pts, ind_data):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_name = os.path.join(save_path, 'img_pt' + str(ind_data) + '.npy')
    np.save(file_name, img_pts)

def generate_img(obj_pts, K, dist_coef, dist_type, save_path):
    cam_ori = [[-10., -10., 0.], [-10., 10, 0.], [-15., -10, 0.],
               [-20., 10, 0.], [-25., -5, 0.], [-30., 10., 0.], 
               [-30., -10., 0.], [-30., 5., 0.]]  # [deg] in the order of XYZ
    
    cam_pos = [[0.1, -0.1, -0.4], [0.1, -0.1, -0.6],
              [0.15, -0.1, -0.5], [0.15, -0.1, -0.6],
              [0.1, -0.15, -0.5]] # [m]
    ind = 1
    for i in cam_ori:
        # Calibrate the camera using OpenCV
        for j in cam_pos:
            # rvec, tvec = conv_pose2Rt(cam_ori, cam_pos)
            rvec, tvec = conv_pose2Rt(i, j)
            if dist_type.startswith('BC'):
                img_pts, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist_coef)
            else:
                img_pts, _ = cv.fisheye.projectPoints(obj_pts, rvec, tvec, K, dist_coef)
            save_img_pts(save_path, img_pts, ind)
            ind += 1
        

if __name__ == '__main__':
    proj_model  = ['P0', 'P1', 'P2', 'P3']
    dist_model = ['BC0', 'BC1', 'BC2', 'BC3', 'BC4', 'KB0', 'KB1', 'KB2', 'KB3', 'KB4']
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    data_path = 'data/synthetic/dataset/'

    ind = 1
    model_ind = []
    synthetic_path = []
    cam_model = []
    K_original = []
    dist_original = []
    obj_pts = gen_chessboard_points(chessboard_pattern)
    for intrinsic_type in proj_model:
        for dist_type in dist_model:
            K = generate_intrinsic_maxtrix(intrinsic_type, img_size)
            dist_coef = generate_dist_coef(dist_type)

            save_path = os.path.join(data_path, 'model_' + str(ind))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            generate_img(obj_pts, K, dist_coef, dist_type, save_path)

            model_ind.append(ind)
            synthetic_path.append(save_path)
            cam_model.append({'dist': dist_type, 'intrinsic': intrinsic_type})
            K_original.append(K)
            dist_original.append(dist_coef)
            ind += 1
    data = {'index': model_ind, 'path': synthetic_path, 'cam_model': cam_model, 'K_original': K_original, 'dist_original': dist_original}  
    df = pd.DataFrame(data)
    df.to_excel(data_path + 'synthetic_data.xlsx', index=False)


