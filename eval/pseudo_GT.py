from camera_calibration import *
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


path = 'data/synthetic/GT_selection/synthetic_5'
df_ori = pd.read_excel(os.path.join(path, 'synthetic_data.xlsx'))
df_AIC_BIC = pd.read_excel(os.path.join(path, 'AIC_BIC.xlsx'))

# Check accuracy for choosing the camera model
img_pts_paths = df_ori['path']
intrinsic_original = df_ori['K_original']
ori_cam_models = df_ori['ori_model']
K_original = df_ori['K_original']
BIC_predicted_models = df_AIC_BIC['AIC_cam_model_predict']
chessboard_pattern = (10, 7)
img_size = (960, 1280)
cam_w, cam_h = img_size[1], img_size[0]
threshold_error = 0.5

BIC_count = 0
mean_error_total = 0
error_count = 0
f_error = 0
ppoint_error = 0
rmse = 0
for i in tqdm(range(len(ori_cam_models))):
    ori_model = eval(ori_cam_models[i])
    BIC_model = eval(BIC_predicted_models[i])
    data_path = img_path = f'data/synthetic/GT_selection/synthetic_40/model_{i+1}'

    if ori_model['dist'] != BIC_model['dist'] or ori_model['intrinsic'] != BIC_model['intrinsic']:
        error_count += 1  
        
        img_pts = load_img_pts(data_path)
        obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
        N_samples = len(obj_pts) * obj_pts[0].shape[0]
        cam_w, cam_h = img_size[1], img_size[0]
        calibrate_flag = CalibrationFlag()
        
        # GT
        dist_type_gt = ori_model['dist']
        intrinsic_type_gt = ori_model['intrinsic']
        flags_gt = calibrate_flag.make_flag(intrinsic_type_gt, dist_type_gt)
        rms_gt, K_gt, dist_coef_gt, rvecs_gt, tvecs_gt = calibrate(obj_pts, img_pts, img_size, 
                                                                   dist_type=dist_type_gt, flags=flags_gt)

        # Prediction
        dist_type_predict = BIC_model['dist']
        intrinsic_type_predict  = BIC_model['intrinsic']
        flags_predict  = calibrate_flag.make_flag(intrinsic_type_predict , dist_type_predict )
        rms_predict , K_predict , dist_coef_predict , rvecs_predict , tvecs_predict  = calibrate(obj_pts, img_pts, 
                                                                                                 img_size, dist_type=dist_type_predict, 
                                                                                                 flags=flags_predict)
        # Calculate repojection error 
        mean_error = find_reproj_error(obj_pts, img_pts, rvecs_predict, tvecs_predict, K_predict, dist_coef_predict, dist_type_predict)
        mean_error = 0
        for i in range(len(obj_pts)):
            reproj_img_points_gt = find_reproject_points(obj_pts[i], rvecs_gt[i], tvecs_gt[i], K_gt, dist_coef_gt, dist_type_gt)
            reproj_img_points_predict = find_reproject_points(obj_pts[i], rvecs_predict[i], tvecs_predict[i], K_predict, dist_coef_predict, dist_type_predict)
            error = cv.norm(reproj_img_points_gt, reproj_img_points_predict, cv.NORM_L2) / np.sqrt(len(reproj_img_points_predict))
            mean_error += error
        if mean_error <= threshold_error:
            BIC_count += 1
        mean_error_total += mean_error
   
    else:
        BIC_count += 1
        # img_pts = load_img_pts(data_path)
        # obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
        # cam_w, cam_h = img_size[1], img_size[0]
        # calibrate_flag = CalibrationFlag()

        # # Prediction
        # dist_type_predict = BIC_model['dist']
        # intrinsic_type_predict  = BIC_model['intrinsic']
        # flags_predict  = calibrate_flag.make_flag(intrinsic_type_predict , dist_type_predict )
        # rms_predict , K_predict , dist_coef_predict , rvecs_predict , tvecs_predict  = calibrate(obj_pts, img_pts, 
        #                                                                                          img_size, dist_type=dist_type_predict, 
        #                                                                                          flags=flags_predict)
        # rmse += rms_predict
        # # Calculate focal length error
        # K_GT = ' '.join(K_original[i].split()).replace('[ ', '[').replace(' ', ', ')
        # K_GT = np.array(eval(K_GT))
        # f_GT = (K_GT[0][0], K_GT[1, 1])
        # f_predict = (K_predict[0][0], K_predict[1, 1])
        # f_error += cv.norm(f_GT, f_predict, cv.NORM_L2)

        # ppoint_GT = (K_GT[0][2], K_GT[1, 2])
        # ppoint_predict = (K_predict[0][2], K_predict[1, 2])
        # ppoint_error += cv.norm(ppoint_predict, ppoint_GT, cv.NORM_L2)
    
print('========================================Error between GT and Prediction=======================================')
print('mean_error = ', mean_error_total / error_count)
print('mean_error_total = ', mean_error_total)
print('BIC_count = ', error_count)

print('========================================Focal Length Error=======================================')
print('mean_f_error = ', f_error / BIC_count)
print('mean_focal_length_error_total = ', f_error)
print('BIC_count = ', BIC_count)

print('========================================Principal Point Error=======================================')
print('mean_ppoint_error = ', ppoint_error / BIC_count)
print('mean_ppoint_error_total = ', ppoint_error)
print('BIC_count = ', BIC_count)

print('========================================RMSE=======================================')
print('rmse = ', rmse / BIC_count)
print('rmse_total = ', rmse)
print('BIC_count = ', BIC_count)

print('========================================BIC Accuracy========================================')
print('Number of correctly predicted models = {}/{}'.format(BIC_count, len(df_AIC_BIC)))
print('Acc = {:.2f}%'.format(BIC_count / len(df_AIC_BIC) * 100))
