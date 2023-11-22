import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.spatial.transform import Rotation
import random
from camera_calibration import CameraModelCombination
import pickle

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

def conv_3param2K(f, image_resolution, cam_type='normal'):
    if cam_type == 'normal':
        splitted_intrinsic = f.split('_')
        if len(splitted_intrinsic) == 1:
            cam_fx = cam_fy = random.randint(500, 1000)
            cx = image_resolution[0] / 2
            cy = image_resolution[1] / 2
        if len(splitted_intrinsic) == 2:
            cam_fx =random.randint(500, 1000)
            cam_fy = random.randint(500, 1000)
            cx = image_resolution[0] / 2
            cy = image_resolution[1] / 2
        if len(splitted_intrinsic) == 3:
            cam_fx = cam_fy = random.randint(500, 1000)
            cx = random.randint(0, image_resolution[0])
            cy = random.randint(0, image_resolution[1])
        if len(splitted_intrinsic) == 4:
            cam_fx =random.randint(500, 1000)
            cam_fy = random.randint(500, 1000)
            cx = random.randint(0, image_resolution[0])
            cy = random.randint(0, image_resolution[1])
        K = np.array([[cam_fx, 0., cx], [0., cam_fy, cy], [0., 0., 1.]])
    else: 
        splitted_intrinsic = f.split('_')
        if len(splitted_intrinsic) == 2:
            cam_fx =random.randint(500, 1000)
            cam_fy = random.randint(500, 1000)
            cx = image_resolution[0] / 2
            cy = image_resolution[1] / 2
            s = 0
        if len(splitted_intrinsic) == 3:
            cam_fx =random.randint(500, 1000)
            cam_fy = random.randint(500, 1000)
            cx = image_resolution[0] / 2
            cy = image_resolution[1] / 2
            s = round(random.uniform(0, 0.09), 2)
        if len(splitted_intrinsic) == 4:
            cam_fx =random.randint(500, 1000)
            cam_fy = random.randint(500, 1000)
            cx = random.randint(0, image_resolution[0])
            cy = random.randint(0, image_resolution[1])
            s = 0
        if len(splitted_intrinsic) == 5:
            cam_fx =random.randint(500, 1000)
            cam_fy = random.randint(500, 1000)
            cx = random.randint(0, image_resolution[0])
            cy = random.randint(0, image_resolution[1])
            s = round(random.uniform(0, 0.09), 2)
        K = np.array([[cam_fx, s, cx], [0., cam_fy, cy], [0., 0., 1.]])

    return K
    

if __name__ == '__main__':
    # Configure a camera
    image_resolution = (1280, 960)
    cam_w, cam_h = (1280, 960)  # [pixel]
    cam_ori = [-10., -10., 0.]  # [deg] in the order of XYZ
    cam_pos = [0.1, -0.1, -0.4] # [m]

    # # Project 3D points
    chessboard_pattern = (10, 7)
    X = gen_chessboard_points(chessboard_pattern)
    
    # K = conv_3param2K(f, image_resolution, cam_type=cam_type)
    # print('K = ', K)
    # rvec, tvec = conv_pose2Rt(cam_ori, cam_pos)
    # cam_dist_coeff = generate_cam_dist_coeff(dist, cam_type=cam_type)
    # x, _ = cv.projectPoints(X, rvec, tvec, K, cam_dist_coeff)
    # with open('synthetic_data/img_points_synthetic.pkl', 'wb') as f:
    #     pickle.dump(x, f)
    
    

    cam_model = CameraModelCombination()
    cam_types = ['normal', 'fisheye']
    ind_data = 1
    for cam_type in cam_types:
        focal_lengths = cam_model.focal_length[cam_type]
        distortions = cam_model.distortion[cam_type]
        for f in focal_lengths:
            for dist in distortions:
                K = conv_3param2K(f, image_resolution, cam_type=cam_type)
                rvec, tvec = conv_pose2Rt(cam_ori, cam_pos)
                cam_dist_coeff = generate_cam_dist_coeff(dist, cam_type=cam_type)
                x, _ = cv.projectPoints(X, rvec, tvec, K, cam_dist_coeff)
                mean, standard_deviation = 0, 0.1
                noise = np.random.normal(mean, standard_deviation, x.shape)
                x_noise = x + noise

                with open('data_synthetic/img_points_synthetic' + str(ind_data) + '.pkl', 'wb') as f:
                    pickle.dump(x_noise, f)
                ind_data += 1
    # Calibrate the camera using OpenCV
    # rms, cv_K, cv_dist_coeff, cv_rvec, cv_tvec = cv.calibrateCamera([X], [x], (cam_w, cam_h), None, None, flags=cv.CALIB_ZERO_TANGENT_DIST)

    # print('\n### Calibrate a camera using OpenCV')
    # with np.printoptions(precision=2, suppress=True):
    #     print(f'* rvec: {cv_rvec[0].flatten()} (Truth: {rvec.flatten()})')
    #     print(f'* t   : {cv_tvec[0].flatten()} (Truth: {tvec.flatten()})')
    #     print(f'* K   : {cv_K.flatten()} (Truth: {K.flatten()})')
    #     print(f'* dist_coeff: {cv_dist_coeff} (Truth: {cam_dist_coeff})')

    # # Visualize the projection on a plane (image)
    # f = plt.figure()
    # plt.plot(x[:,0,0], x[:,0,1], 'k.')
    # plt.axis('equal')
    # plt.xlim([0, cam_w])
    # plt.ylim([cam_h, 0])
    # plt.show()