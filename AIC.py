import numpy as np
import pandas as pd
from camera_calibration import CalibrationFlag
from generate_synthetic_data import CameraModelCombination
import cv2 as cv
import math

def gen_chessboard_points(chessboard_pattern, cellsize=0.025, dtype=np.float32):
    '''Generate 3D points with regular interval on a plane (a.k.a. chessboard).'''
    obj_pts = cellsize * np.array([(c, r, 0) for r in range(chessboard_pattern[1]) for c in range(chessboard_pattern[0])], dtype=dtype)
    row, col = obj_pts.shape
    obj_pts = obj_pts.reshape(row, 1, col)
    return obj_pts


def error_cal(obj_points, img_points, rvecs, tvecs, K, dist, cam_type):
    proj_error = 0
    # Reprojection
    if cam_type == 'fisheye':
        reproj_img_points, _ = cv.fisheye.projectPoints(obj_points, rvecs, tvecs, K, dist)
    else:
        reproj_img_points, _ = cv.projectPoints(obj_points, rvecs, tvecs, K, dist)
    
    for i, j in zip(img_points, reproj_img_points):
        proj_error += pow(i[0][0] - j[0][0], 2) + pow(i[0][1] - j[0][1], 2)
    return proj_error


def cal_aic(error, dist, f, sigma):
    if dist == 'no':
        num_para = len(f.split('_'))
    else:
        num_para = len(f.split('_')) + len(dist.split('_'))

    return 2 * num_para + error / pow(sigma, 2)

def new_cal_aic(error, dist, f, N=70):
    if dist == 'no':
        num_para = len(f.split('_'))
    else:
        num_para = len(f.split('_')) + len(dist.split('_'))

    return 2 * num_para + N * np.log(pow(error, 2))


def cal_bic(error, dist, f, sigma, N = 70):
    if dist == 'no':
        num_para = len(f.split('_'))
    else:
        num_para = len(f.split('_')) + len(dist.split('_'))

    return error / pow(sigma, 2) +  math.log(N) * num_para


if __name__ == '__main__':
    data = pd.read_excel('data_synthetic/synthetic.xlsx')
    file_path = data['file_path']
    generated_cam_model = data['cam_model']
    cam_types = ['normal', 'fisheye']
    cam_model = CameraModelCombination()
    chessboard_pattern = (10, 7)
    (cam_w, cam_h) = (1280, 960)
    sigma = 0.01
    X = gen_chessboard_points(chessboard_pattern)

    best_camera_model = []
    AIC_score =[]
    for img in file_path:
        x_noise =np.load(img)
        best_AIC = 100000
        best_cam_type = ''
        best_f = ''
        best_dist = ''
        for cam_type in cam_types:
            frame = []
            focal_lengths = cam_model.focal_length[cam_type]
            distortions = cam_model.distortion[cam_type]
            for f in focal_lengths:
                result = []
                for dist in distortions: 
                    cali_flag = CalibrationFlag()
                    flags_calibrate = cali_flag.make_cali_flag(focal_length=f, distortion=dist, cam_type=cam_type)
                    try:
                        if cam_type == 'fisheye':
                            rms, cv_K, cv_dist_coeff, cv_rvec, cv_tvec = cv.fisheye.calibrate([X], [x_noise], (cam_w, cam_h), None, None, flags=flags_calibrate)                 
                        else:
                            rms, cv_K, cv_dist_coeff, cv_rvec, cv_tvec = cv.calibrateCamera([X], [x_noise], (cam_w, cam_h), None, None, flags=flags_calibrate)
                        proj_error = error_cal(X, x_noise, cv_rvec[0], cv_tvec[0], cv_K, cv_dist_coeff, cam_type=cam_type)
                        AIC = new_cal_aic(rms, dist, f)
                    except:
                        AIC = 10000
                    if AIC < best_AIC:
                        best_AIC = AIC 
                        best_cam_type = cam_type
                        best_f = f
                        best_dist = dist
        best_camera_model.append({'type' :best_cam_type, 'f': best_f, 'dist': best_dist})
        AIC_score.append(best_AIC)
    
    data['best_camera_model'] = best_camera_model
    data['AIC'] = AIC_score
    data.to_excel('data_synthetic/AIC_results.xlsx')