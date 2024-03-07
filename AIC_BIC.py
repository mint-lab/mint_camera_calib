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

def caculate_model_score(train_error, test_error, num_para, num_img, sampling_type):
    # return -np.log(test_error_weight / test_error + train_error_weight / train_error + ratio_weight * train_test_ratio)
    # return np.log(train_error + test_error + pow(train_error, 2) / test_error + num_para * test_error)
    # return train_error / test_error + test_error / train_error
    # return abs(train_error - test_error) * (train_error**2 + test_error**2)**(0.5) * (num_para + 2)**(0.1)
    # return np.log(train_error + test_error + pow(train_error, 2) / test_error + num_para * test_error)
    if sampling_type == 'random':
        num_train = 49
        num_test = 21
    elif sampling_type == 'extrapolar':
        num_train = 30
        num_test = 40
    elif sampling_type == 'interpolar':
        num_train = 40
        num_test = 30
    return num_train * num_img * np.log(pow(train_error, 2)) + num_test * num_img * np.log(pow(test_error, 2)) + num_para * np.log10((num_train + num_test) * num_img)

def find_best_model_AIC(RMSE_BC, RMSE_KB, N_samples, proj_model_BC, proj_model_KB, dist_model_BC, dist_model_KB):
    num_para_BC = get_num_para_BC()
    num_para_KB = get_num_para_KB()

    AIC_BC = cal_AIC(RMSE_BC, N_samples, num_para_BC)
    AIC_KB = cal_AIC(RMSE_KB, N_samples, num_para_KB)

    AIC_df_BC = pd.DataFrame(AIC_BC, index=proj_model_BC, columns=dist_model_BC)
    AIC_df_KB = pd.DataFrame(AIC_KB, index=proj_model_KB, columns=dist_model_KB)
    AIC_df = pd.concat([AIC_df_BC, AIC_df_KB], axis=1)
    _, min_AIC_intrinsic, min_AIC_dist= find_df_min_value(AIC_df)

    return min_AIC_intrinsic, min_AIC_dist

def find_best_model_BIC(RMSE_BC, RMSE_KB, N_samples, proj_model_BC, proj_model_KB, dist_model_BC, dist_model_KB):
    num_para_BC = get_num_para_BC()
    num_para_KB = get_num_para_KB()

    BIC_BC = cal_BIC(RMSE_BC, N_samples, num_para_BC)
    BIC_KB = cal_BIC(RMSE_KB, N_samples, num_para_KB)

    BIC_df_BC = pd.DataFrame(BIC_BC, index=proj_model_BC, columns=dist_model_BC)
    BIC_df_KB = pd.DataFrame(BIC_KB, index=proj_model_KB, columns=dist_model_KB)
    BIC_df = pd.concat([BIC_df_BC, BIC_df_KB], axis=1)
    _, min_BIC_intrinsic, min_BIC_dist= find_df_min_value(BIC_df)

    return min_BIC_intrinsic, min_BIC_dist


if __name__ == '__main__':
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    data_path = 'data/synthetic_new'
    proj_model_BC = ['P0', 'P1', 'P2', 'P3']
    dist_model_BC = ['BC0', 'BC1', 'BC2', 'BC3']
    proj_model_KB = ['P1', 'P3']
    dist_model_KB =['KB0', 'KB1', 'KB2']

    df = pd.read_excel(os.path.join(data_path, 'synthetic_data.xlsx'))
    ori_models = df['ori_model']
    paths = df['path']

    acc = 0
    best_model_AIC, best_model_BIC = [], []
    obj_pts = generate_obj_points(chessboard_pattern, len(load_img_pts(paths[0])))
    for path, ori_model in tqdm(zip(paths, ori_models)):
        img_pts = load_img_pts(path)
        ori_model = eval(ori_model)
        N_samples = len(obj_pts) * obj_pts[0].shape[0]

        RMSE_BC = cali_error_process(obj_pts, img_pts, proj_model_BC, dist_model_BC, img_size)
        RMSE_KB = cali_error_process(obj_pts, img_pts, proj_model_KB, dist_model_KB, img_size)
        np.save(os.path.join(path, 'rms_bc.npy'), RMSE_BC)
        np.save(os.path.join(path, 'rms_kb.npy'), RMSE_KB)

        AIC_best_intrinsic, AIC_best_dist= find_best_model_AIC(RMSE_BC, RMSE_KB, N_samples, 
                                                                proj_model_BC, proj_model_KB, 
                                                                dist_model_BC, dist_model_KB)
        best_model_AIC.append({'dist': AIC_best_dist, 'intrinsic': AIC_best_intrinsic})
        

            
        BIC_best_intrinsic, BIC_best_dist= find_best_model_BIC(RMSE_BC, RMSE_KB, N_samples, 
                                                                proj_model_BC, proj_model_KB, 
                                                                dist_model_BC, dist_model_KB)
        best_model_BIC.append({'dist': BIC_best_dist, 'intrinsic': BIC_best_intrinsic})
        
    df['AIC_cam_model_predict'] = best_model_AIC
    df['BIC_cam_model_predict'] = best_model_BIC
    df.to_excel(os.path.join(data_path, 'AIC_BIC.xlsx'), index=False)