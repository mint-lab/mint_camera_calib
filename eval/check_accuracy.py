from camera_calibration import *
import pandas as pd
import regex as re
from tqdm import tqdm


path = 'data/synthetic/GT_selection/synthetic_5'
df_ori = pd.read_excel(os.path.join(path, 'synthetic_data.xlsx'))
df_AIC_BIC = pd.read_excel(os.path.join(path, 'AIC_BIC.xlsx'))

# Check accuracy for choosing the camera model
img_pts_paths = df_ori['path']
intrinsic_original = df_ori['K_original']
ori_cam_models = df_ori['ori_model']
AIC_predicted_models = df_AIC_BIC['AIC_cam_model_predict']
BIC_predicted_models = df_AIC_BIC['BIC_cam_model_predict']

AIC_count = 0
BIC_count = 0
for ori_model, AIC_model, BIC_model in zip(ori_cam_models, AIC_predicted_models, BIC_predicted_models):
    ori_model = eval(ori_model)
    AIC_model = eval(AIC_model)
    BIC_model = eval(BIC_model)

    if ori_model['dist'] == AIC_model['dist'] and ori_model['intrinsic'] == AIC_model['intrinsic']:
    # if ori_model['dist'] == AIC_model['dist']:
        AIC_count += 1

    if ori_model['dist'] == BIC_model['dist'] and ori_model['intrinsic'] == BIC_model['intrinsic']:
    # if ori_model['dist'] == BIC_model['dist']:
        BIC_count += 1

print('========================================AIC Accuracy========================================')
print('Number of correctly predicted models = {}/{}'.format(AIC_count, len(df_AIC_BIC)))
print('Acc = {:.2f}%'.format(AIC_count / len(df_AIC_BIC) * 100))

print('========================================BIC Accuracy========================================')
print('Number of correctly predicted models = {}/{}'.format(BIC_count, len(df_AIC_BIC)))
print('Acc = {:.2f}%'.format(BIC_count / len(df_AIC_BIC) * 100))

# Check RMSE for Focal Length, Principal Point, Reprojection Error
chessboard_pattern = (10, 7)
img_size = (960, 1280)
f_error = []
principal_pt_error = []   
rmse = []

for i in tqdm(range(len(df_ori))):
    img_path = f'data/synthetic/GT_selection/synthetic_40/model_{i+1}'
    img_pts = load_img_pts(img_path)
    obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
    calibrate_flag = CalibrationFlag()
    cam_type = eval(AIC_predicted_models[i])
    intrinsic_type = cam_type['intrinsic']
    dist_type = cam_type['dist']
    flags = calibrate_flag.make_flag(intrinsic_type, dist_type)
    
    if dist_type.startswith('BC'):
        rms, K, dist_coef, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_pts, img_size[::-1], cameraMatrix=None, distCoeffs=None, flags=flags)
    else:
        rms, K, dist_coef, rvecs, tvecs = cv.fisheye.calibrate(obj_pts, img_pts, img_size[::-1], K=None, D=None, flags=flags)

    fx_predict = K[0][0]
    fy_predict = K[1][1]
    cx_predict = K[0][2]
    cy_predict = K[1][2]

    tmp = intrinsic_original[i]
    # tmp = np.array(eval(tmp.replace('\n', '').replace(' ', ' ').replace('   ', ' ').replace('[  ', '[').replace(' ', ',')))
    tmp = re.sub(r'\s+', ' ', tmp)
    tmp = re.sub(r'\[\s+', '[', tmp)
    tmp = np.array(eval(tmp.replace('\n', '').replace(' ', ',')))
    fx_original = tmp[0][0]
    fy_original = tmp[1][1]
    cx_original = tmp[0][2]
    cy_original = tmp[1][2]
    
    f_error.append(pow(fx_predict - fx_original, 2) + pow(fy_predict - fy_original, 2))
    principal_pt_error.append(pow(cx_predict - cx_original, 2) + pow(cy_predict - cy_original, 2))
    rmse.append(rms)
    
f_rmse = np.sqrt(np.mean(np.array(f_error)))
principal_pt_rmse = np.sqrt(np.mean(np.array(principal_pt_error)))
reproj_rmse = np.mean(np.array(rmse))

print('==============')
print('f_rmse', f_rmse)
print('principal_pt_rmse', principal_pt_rmse)
print('reproj_rmse', reproj_rmse)