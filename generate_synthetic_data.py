import numpy as np
import cv2 as cv
import pandas as pd
import random
from scipy.spatial.transform import Rotation
from camera_calibration import CameraModelCombination


def random_principal_point(img_size):
    cx = random.randint(img_size[1] / 2 + 10, img_size[1] / 2 + 10)
    cy = random.randint(img_size[0] / 2 - 10, img_size[0] / 2 + 10)

    return cx, cy   

def gen_chessboard_points(chessboard_pattern, cellsize=0.025, dtype=np.float32):
    '''Generate 3D points with regular interval on a plane (a.k.a. chessboard).'''
    obj_pts = cellsize * np.array([(c, r, 0) for r in range(chessboard_pattern[1]) for c in range(chessboard_pattern[0])], dtype=dtype)
    row, col = obj_pts.shape
    obj_pts = obj_pts.reshape(row, 1, col)
    return obj_pts

def conv_pose2Rt(ori_xyz, pos_xyz, degrees=True):
    '''Convert the given camera pose to the rotation vector (rvec) and translation vector (tvec).'''
    rvec = -Rotation.from_euler('zyx', ori_xyz[::-1], degrees=degrees).as_rotvec() # '-' is transpose for 'rvec'.
    tvec = -Rotation.from_rotvec(rvec).apply(pos_xyz)
    return rvec.ravel(), tvec.ravel()

def generate_cam_dist_coeff(dist, cam_type='normal'):
    random_range = (0, 0.09)
    num_coefficients = 5 if cam_type == 'normal' else 4
    cam_dist_coeff = np.round([0.] * num_coefficients)
    
    if dist != 'no':
        for i in range(len(dist.split('_'))):
            cam_dist_coeff[i] = round(random.uniform(random_range[0], random_range[1]), 2) 
    return cam_dist_coeff

def conv_3param2K(f, img_size, cam_type='normal'):
    if cam_type == 'normal':  
        if f == 'f':
            cam_fx = cam_fy = random.randint(800, 1000)
            cx = img_size[1] / 2
            cy = img_size[0] / 2
        if f == 'fx_fy':
            cam_fx =random.randint(800, 1000)
            cam_fy = random.randint(800, 1000)
            cx = img_size[1] / 2
            cy = img_size[0] / 2
        if f == 'f_cx_cy':
            cam_fx = cam_fy = random.randint(800, 1000)
            cx, cy = random_principal_point(img_size)
        if f == 'fx_fy_cx_cy':
            cam_fx = random.randint(800, 1000)
            cam_fy = random.randint(800, 1000)
            cx, cy = random_principal_point(img_size)
        K = np.array([[cam_fx, 0., cx], [0., cam_fy, cy], [0., 0., 1.]])
    else: 
        if f == 'fx_fy':
            cam_fx =random.randint(800, 1000)
            cam_fy = random.randint(800, 1000)
            cx = img_size[1] / 2
            cy = img_size[0] / 2
            s = 0
        if f == 'fx_fy_s':
            cam_fx =random.randint(800, 1000)
            cam_fy = random.randint(800, 1000)
            cx = img_size[1] / 2
            cy = img_size[0] / 2
            s = round(random.uniform(0, 0.09), 2)
        if f == 'fx_fy_cx_cy':
            cam_fx =random.randint(800, 1000)
            cam_fy = random.randint(800, 1000)
            cx, cy = random_principal_point(img_size)
            s = 0
        if f == 'fx_fy_cx_cy_s':
            cam_fx =random.randint(800, 1000)
            cam_fy = random.randint(800, 1000)
            cx, cy = random_principal_point(img_size)
            s = round(random.uniform(0, 0.09), 2)
        K = np.array([[cam_fx, s, cx], [0., cam_fy, cy], [0., 0., 1.]])

    return K

def add_noise(x):
    mean, standard_deviation = 0, 0.01
    noise = np.random.normal(mean, standard_deviation, x.shape)
    x = x + noise
    x = np.float32(np.round(x, decimals = 4))

    return x

def project_points(X, rvec, tvec, K, cam_dist_coeff, cam_type='normal'):
    if cam_type == 'fisheye':
        x, _ = cv.fisheye.projectPoints(X, rvec, tvec, K, cam_dist_coeff)
    else:
        x, _ = cv.projectPoints(X, rvec, tvec, K, cam_dist_coeff)

    return x

if __name__ == '__main__':
    # Configure a camera
    cam_w, cam_h = (1280, 960)  # [pixel]
    img_size = (cam_h, cam_w)
    cam_ori = [-10., -10., 0.]  # [deg] in the order of XYZ
    cam_pos = [0.1, -0.1, -0.4] # [m]

    # Project 3D points
    chessboard_pattern = (10, 7)
    X = gen_chessboard_points(chessboard_pattern)
    
    cam_model = CameraModelCombination()
    cam_types = ['normal', 'fisheye']
    ind_data = 1
    index = []
    file_paths = []
    camera_model = []
    distortion = []
    intrinsic = []
    for cam_type in cam_types:
        focal_lengths = cam_model.focal_length[cam_type]
        distortions = cam_model.distortion[cam_type]
        for f in focal_lengths:
            for dist in distortions:
                for _ in range(10):
                    K = conv_3param2K(f, img_size, cam_type=cam_type)
                    rvec, tvec = conv_pose2Rt(cam_ori, cam_pos)
                    cam_dist_coeff = generate_cam_dist_coeff(dist, cam_type=cam_type)
                    x = project_points(X, rvec, tvec, K, cam_dist_coeff, cam_type=cam_type)
                    x = add_noise(x)
                    file_name = 'data_synthetic/synthetic_img_pt' + str(ind_data) + '.npy'
                    np.save(file_name, x) 
                    index.append(ind_data)
                    file_paths.append(file_name)
                    camera_model.append({'type' :cam_type, 'f': f, 'dist': dist})
                    intrinsic.append(K.flatten())
                    distortion.append(cam_dist_coeff)
                    ind_data += 1

    df = pd.DataFrame({'ind' :index, 'file_path': file_paths, 'intrinsic': intrinsic, 'distortion': distortion,  "cam_model": camera_model})
    df.to_excel('data_synthetic/synthetic.xlsx', index=False)