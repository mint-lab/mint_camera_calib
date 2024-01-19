import re
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2 as cv
from camera_calibration import calibrate, generate_obj_points, load_img_pts, CalibrationFlag

path = 'data/synthetic'
sampling_type = 'interpolar'
df_ori = pd.read_excel(os.path.join(path, 'synthetic_data.xlsx'))
df_AIC = pd.read_excel(os.path.join(path, 'AIC.xlsx'))
df_BIC = pd.read_excel(os.path.join(path, 'BIC.xlsx'))
df_proposal = pd.read_excel(os.path.join(path, sampling_type + '.xlsx'))

# Check accuracy for choosing the distortion model
AIC_count = 0
BIC_count = 0
proposal_count = 0
img_pts_paths = df_ori['path']
intrinsic_original = df_ori['K_original']
ori_cam_model= df_ori['ori_model']
predicted_model_AIC = df_AIC['AIC_cam_model_predict']
predicted_model_BIC = df_BIC['BIC_cam_model_predict']
predicted_model_proposal = df_proposal['proposal_cam_model_predict']

ind = 1
for ori_model, AIC_model, BIC_model, prosal_model in zip(ori_cam_model, predicted_model_AIC, predicted_model_BIC, predicted_model_proposal):
    ori_model = eval(ori_model)
    AIC_model = eval(AIC_model)
    BIC_model = eval(BIC_model)
    prosal_model = eval(prosal_model) 

    if ori_model['dist'] == AIC_model['dist']:
        AIC_count += 1
    if ori_model['dist'] == BIC_model['dist']:
        BIC_count += 1
    if ori_model['dist'] == prosal_model['dist']:
        proposal_count += 1

print(AIC_count)
print(AIC_count / len(df_AIC) * 100)
print('=============================')
print(BIC_count)
print(BIC_count / len(df_BIC) * 100)
print('=============================')
print(proposal_count)
print(proposal_count / len(df_proposal) * 100)

# Check RMSE for Focal Length, Principal Point, Reprojection Error
# chessboard_pattern = (10, 7)
# img_size = (960, 1280)
# f_error = []
# principal_pt_error = []
# rmse = []
# proposal_cam_model_predict = df_proposal['proposal_cam_model_predict']
# for i in tqdm(range(len(df_ori))):
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