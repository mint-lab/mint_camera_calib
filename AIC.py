import numpy as np
import pandas as pd
import glob
import cv2 as cv
import os
from tqdm import tqdm
from camera_calibration import  CalibrationFlag, CameraModelCombination


def initilize_obj_points(board_pattern, len_img_points):
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
    row, col = obj_points[0].shape
    obj_points = [x.reshape(row, 1, col) for x in obj_points]
    return obj_points

def load_img(input_path):
    files = glob.glob(input_path + '*.npy')
    img_select = [np.load(file) for file in files]
    return img_select

def gen_chessboard_points(chessboard_pattern, cellsize=0.025, dtype=np.float32):
    '''Generate 3D points with regular interval on a plane (a.k.a. chessboard).'''
    return cellsize * np.array([(c, r, 0) for r in range(chessboard_pattern[1]) for c in range(chessboard_pattern[0])], dtype=dtype)

def error_cal(obj_points, img_points, rvecs, tvecs, K, dist, cam_type):
    # Reprojection
    proj_error = 0
    for i in range(len(obj_points)):
        if cam_type == 'fisheye':
            reproj_img_points, _ = cv.fisheye.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        else:
            reproj_img_points, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        
        for i, j in zip(img_points[i], reproj_img_points):
            proj_error += pow(i[0][0] - j[0][0], 2) + pow(i[0][1] - j[0][1], 2)
            # proj_error += i[0][0] - j[0][0] + i[0][1] - j[0][1]

    return proj_error

def cal_aic(RMSE, dist, f, N):
    if dist == 'no':
        num_para = len(f.split('_'))
    else:
        num_para = len(f.split('_')) + len(dist.split('_'))

    return N * np.log(pow(RMSE, 2)) + 2 * num_para

def cal_bic(RMSE, dist, f, N):
    if dist == 'no':
        num_para = len(f.split('_'))
    else:
        num_para = len(f.split('_')) + len(dist.split('_'))

    return N * np.log(pow(RMSE, 2)) + num_para * np.log10(N)

def add_noise(x, mean=0, standard_deviation=0.01):
    noise = np.random.normal(mean, standard_deviation, x.shape)
    x = x + noise
    x = np.float32(np.round(x, decimals = 4))

    return x

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    array = (array - min_val) / (max_val - min_val)

    return array

if __name__ == '__main__':
    chessboard_pattern = (10, 7)
    (cam_w, cam_h) = (1280, 960)
    load_path = 'data/synthetic/normal_40_random/'
    save_path = 'results/synthetic/normal_40_random/AIC/'

    x = load_img(load_path)
    # x = [add_noise(i, standard_deviation=0.03) for i in x]
    X = initilize_obj_points(chessboard_pattern, len(x))
    num_samples = chessboard_pattern[0] * chessboard_pattern[1] * len(X)

    cam_types = ['normal', 'fisheye']
    cam_model = CameraModelCombination()
    AIC_all = []
    BIC_all = []
    RMSE_all = []
    for cam_type in tqdm(cam_types): 
        intrinsics = cam_model.intrinsic[cam_type]
        distortions = cam_model.distortion[cam_type]
        AICs = []
        BICs = []
        RMSEs = []
        for intrinsic_type in tqdm(intrinsics):
            AIC = []
            BIC = []
            RMSE = []
            for dist_type in distortions: 
                cali_flag = CalibrationFlag()
                flags_calibrate = cali_flag.make_cali_flag(intrinsic_type=intrinsic_type, dist_type=dist_type, cam_type=cam_type)
                if cam_type == 'fisheye':
                    rms, cv_K, cv_dist_coeff, cv_rvec, cv_tvec = cv.fisheye.calibrate(X, x, (cam_w, cam_h), None, None, flags=flags_calibrate)
                else:
                    rms, cv_K, cv_dist_coeff, cv_rvec, cv_tvec = cv.calibrateCamera(X, x, (cam_w, cam_h), None, None, flags=flags_calibrate)                   
                AIC.append(cal_aic(rms, dist_type, intrinsic_type, num_samples))
                BIC.append(cal_bic(rms, dist_type, intrinsic_type, num_samples))
                RMSE.append(rms)
            AICs.append(AIC)
            BICs.append(BIC)
            RMSEs.append(RMSE)
        AIC_all.append(AICs)
        BIC_all.append(BICs)
        RMSE_all.append(RMSEs)

    AIC_all= np.append(AIC_all[0], AIC_all[1], axis=1)
    BIC_all = np.append(BIC_all[0], BIC_all[1], axis=1)
    RMSE_all = np.append(RMSE_all[0], RMSE_all[1], axis=1)
    
    index = ['P0', 'P1', 'P2', 'P3']
    columns = ['BC0', 'BC1', 'BC2', 'BC3', 'BC4', 'KB0', 'KB1', 'KB2', 'KB3', 'KB4']
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    df_AIC = pd.DataFrame(AIC_all, index=index, columns=columns)
    df_AIC.to_excel(save_path + 'AIC.xlsx')
    np.save(save_path + 'AIC.npy', AIC_all)

    df_BIC = pd.DataFrame(BIC_all, index=index, columns=columns)
    df_BIC.to_excel(save_path + 'BIC.xlsx')
    np.save(save_path + 'BIC.npy', BIC_all)
    
    df_RMSE = pd.DataFrame(RMSE_all, index=index, columns=columns)
    df_RMSE.to_excel(save_path + 'RMSE.xlsx')
    np.save(save_path + 'RMSE', RMSE_all)