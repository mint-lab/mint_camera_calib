from camera_calibration import *

def get_num_para_BC():
    dist_model_num_para = [0, 1, 2, 4]
    proj_model_num_para = [1, 2, 3, 4]
    num_para = np.array([[dist + f for dist in dist_model_num_para] for f in proj_model_num_para])
    return num_para

def get_num_para_KB():
    dist_model_num_para = [0, 1, 2]
    proj_model_num_para = [2, 4]
    num_para = np.array([[dist + f for dist in dist_model_num_para] for f in proj_model_num_para])
    return num_para

def cal_AIC(RMSE, N_samples, num_para):  
    return N_samples * np.log(pow(RMSE, 2)) + 2 * num_para

def cal_BIC(RMSE, N_samples, num_para):
    return N_samples * np.log(pow(RMSE, 2)) + num_para * np.log10(N_samples)

def find_best_model(RMSE_BC, RMSE_KB, proj_model_BC, proj_model_KB, dist_model_BC, dist_model_KB, N_samples, criteria_type):
    num_para_BC = get_num_para_BC()
    num_para_KB = get_num_para_KB()

    BC = cal_AIC(RMSE_BC, N_samples, num_para_BC) if criteria_type == 'aic' else cal_BIC(RMSE_BC, N_samples, num_para_BC)
    KB = cal_AIC(RMSE_KB, N_samples, num_para_KB) if criteria_type == 'aic' else cal_BIC(RMSE_KB, N_samples, num_para_KB)

    df_BC = pd.DataFrame(BC, index=proj_model_BC, columns=dist_model_BC)
    df_KB = pd.DataFrame(KB, index=proj_model_KB, columns=dist_model_KB)
    df = pd.concat([df_BC, df_KB], axis=1)
    _, min_intrinsic, min_dist= find_df_min_value(df)

    return min_intrinsic, min_dist


if __name__ == '__main__':
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    data_path = 'data/synthetic'
    proj_model_BC = ['P0', 'P1', 'P2', 'P3']
    dist_model_BC = ['BC0', 'BC1', 'BC2', 'BC3']
    proj_model_KB = ['P1', 'P3']
    dist_model_KB =['KB0', 'KB1', 'KB2']

    df = pd.read_excel(os.path.join(data_path, 'synthetic_data.xlsx'))
    paths = df['path']

    best_model_AIC, best_model_BIC = [], []
    obj_pts = generate_obj_points(chessboard_pattern, len(load_img_pts(paths[0])))
    N_samples = len(obj_pts) * obj_pts[0].shape[0]
    for path in tqdm(paths):
        img_pts = load_img_pts(path)

        RMSE_BC = cali_error_process(obj_pts, img_pts, proj_model_BC, dist_model_BC, img_size)
        RMSE_KB = cali_error_process(obj_pts, img_pts, proj_model_KB, dist_model_KB, img_size)
        np.save(os.path.join(path, 'rms_bc.npy'), RMSE_BC)
        np.save(os.path.join(path, 'rms_kb.npy'), RMSE_KB)

        AIC_best_intrinsic, AIC_best_dist= find_best_model(RMSE_BC, RMSE_KB, 
                                                            proj_model_BC, proj_model_KB, 
                                                            dist_model_BC, dist_model_KB, 
                                                            N_samples, criteria_type='aic')
        best_model_AIC.append({'dist': AIC_best_dist, 'intrinsic': AIC_best_intrinsic})
        
        BIC_best_intrinsic, BIC_best_dist= find_best_model(RMSE_BC, RMSE_KB, 
                                                            proj_model_BC, proj_model_KB, 
                                                            dist_model_BC, dist_model_KB,
                                                            N_samples, criteria_type='bic')
        best_model_BIC.append({'dist': BIC_best_dist, 'intrinsic': BIC_best_intrinsic})
        
    df['AIC_cam_model_predict'] = best_model_AIC
    df['BIC_cam_model_predict'] = best_model_BIC
    df.to_excel(os.path.join(data_path, 'AIC_BIC.xlsx'), index=False)