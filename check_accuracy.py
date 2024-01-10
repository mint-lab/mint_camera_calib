import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2 as cv
from camera_calibration import calibrate, generate_obj_points, load_img_pts, CalibrationFlag

df = pd.read_excel('data/synthetic/dataset_noise_1/accuracy_extrapolar_noise_1.xlsx')
df1 = pd.read_excel('data/synthetic/dataset_noise_1/accuracy_random_noise_1.xlsx')
img_pts_paths = df['path']
intrinsic_original = df['K_original']

# Check accuracy
AIC_count = 0
BIC_count = 0
proposal_count = 0
cam_model_ogiginal = df['cam_model']
AIC_cam_model_predicted = df['AIC_cam_model_predict']
BIC_cam_model_predicted = df['BIC_cam_model_predict']
proposal_cam_model_predict = df['proposal_cam_model_predict']
for i, j, k, m in zip(cam_model_ogiginal, AIC_cam_model_predicted, BIC_cam_model_predicted, proposal_cam_model_predict):
    i = eval(i)
    j = eval(j)
    k = eval(k)
    m = eval(m)
    if i['dist'] == j['dist']:
        AIC_count += 1
    if i['dist'] == k['dist']:
        BIC_count += 1
    if i['dist'] == m['dist']:
        proposal_count += 1
print(AIC_count)
print(AIC_count / 400 * 100)
print('=============================')
print(BIC_count)
print(BIC_count / 400 * 100)
print('=============================')
print(proposal_count)
print(proposal_count / 400 * 100)

# Check Principal Point, Focal Length, Reprojection Error
# chessboard_pattern = (10, 7)
# img_size = (960, 1280)
# f_error = []
# principal_pt_error = []
# rmse = []
# proposal_cam_model_predict = df1['proposal_cam_model_predict']
# for i in tqdm(range(len(df))):
#     img_pts = load_img_pts(img_pts_paths[i])
#     obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
#     calibrate_flag = CalibrationFlag()
#     cam_type = eval(proposal_cam_model_predict[i])
#     intrinsic_type = cam_type['intrinsic']
#     dist_type = cam_type['dist']
#     flags = calibrate_flag.make_flag(intrinsic_type, dist_type)
    
#     if dist_type.startswith('BC'):
#         rms, K, dist_coef, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_pts, img_size[::-1], cameraMatrix=None, distCoeffs=None, flags=flags)
#     else:
#         rms, K, dist_coef, rvecs, tvecs = cv.fisheye.calibrate(obj_pts, img_pts, img_size[::-1], K=None, D=None, flags=flags)

#     fx_predict = K[0][0]
#     fy_predict = K[1][1]
#     cx_predict = K[0][2]
#     cy_predict = K[1][2]

#     tmp = intrinsic_original[i]
#     # tmp = np.array(eval(tmp.replace('\n', '').replace(' ', ' ').replace('   ', ' ').replace('[  ', '[').replace(' ', ',')))
#     tmp = re.sub(r'\s+', ' ', tmp)
#     tmp = re.sub(r'\[\s+', '[', tmp)
#     tmp = np.array(eval(tmp.replace('\n', '').replace(' ', ',')))
#     fx_original = tmp[0][0]
#     fy_original = tmp[1][1]
#     cx_original = tmp[0][2]
#     cy_original = tmp[1][2]
    
#     f_error.append(pow(fx_predict - fx_original, 2) + pow(fy_predict - fy_original, 2))
#     principal_pt_error.append(pow(cx_predict - cx_original, 2) + pow(cy_predict - cy_original, 2))
#     rmse.append(rms)
    
# f_rmse = np.sqrt(np.mean(np.array(f_error)))
# principal_pt_rmse = np.sqrt(np.mean(np.array(principal_pt_error)))
# reproj_rmse = np.mean(np.array(rmse))

# print('==============')
# print('f_rmse', f_rmse)
# print('principal_pt_rmse', principal_pt_rmse)
# print('reproj_rmse', reproj_rmse)