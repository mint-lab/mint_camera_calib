from camera_calibration import *


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

def find_best_model(RMSE_BC, RMSE_KB, N_samples, proj_model_BC, proj_model_KB, dist_model_BC, dist_model_KB):
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
    data_path = 'data/synthetic_new_5'
    proj_model_BC = ['P0', 'P1', 'P2', 'P3']
    dist_model_BC = ['BC0', 'BC1', 'BC2', 'BC3']
    proj_model_KB = ['P1', 'P3']
    dist_model_KB =['KB0', 'KB1', 'KB2']

    df = pd.read_excel(os.path.join(data_path, 'synthetic_data.xlsx'))
    img_pts_paths = df['path']
    cam_model = df['ori_model']

    sampling_type = 'random'
    obj_pts = generate_obj_points(chessboard_pattern, len(load_img_pts(img_pts_paths[0])))
    num_img = len(obj_pts)
    num_para_BC = get_num_para_BC()
    num_para_KB = get_num_para_KB()
    best_model_proposal = []
    for img_pts_path in tqdm(img_pts_paths):
        img_pts = load_img_pts(img_pts_path)

        # Proposal'
        data = DataSampling(obj_pts, img_pts, chessboard_pattern)
        obj_pts_train, img_pts_train, obj_pts_test, img_pts_test = data.split_data(sampling_type=sampling_type)
        # BC
        BC_train_error, BC_test_error = train_test_error_process(obj_pts_train, img_pts_train, obj_pts_test,
                                                           img_pts_test, proj_model_BC, dist_model_BC, img_size)
        np.save(os.path.join(img_pts_path, 'BC_train_error_' + sampling_type + '.npy'), BC_train_error)
        np.save(os.path.join(img_pts_path, 'BC_test_error_' + sampling_type + '.npy'), BC_test_error)
        # KB
        KB_train_error, KB_test_error = train_test_error_process(obj_pts_train, img_pts_train, obj_pts_test,
                                                           img_pts_test, proj_model_KB, dist_model_KB, img_size)
        np.save(os.path.join(img_pts_path, 'KB_train_error_' + sampling_type + '.npy'), KB_train_error)
        np.save(os.path.join(img_pts_path, 'KB_test_error_' + sampling_type + '.npy'), KB_test_error)
        
        # Score
        model_score_BC = caculate_model_score(BC_train_error, BC_test_error, num_para_BC, num_img, sampling_type)
        score_df_BC = pd.DataFrame(model_score_BC, index=proj_model_BC, columns=dist_model_BC)
        model_score_KB = caculate_model_score(KB_train_error, KB_test_error, num_para_KB, num_img, sampling_type)
        score_df_KB = pd.DataFrame(model_score_KB, index=proj_model_KB, columns=dist_model_KB)
        score_df = pd.concat([score_df_BC, score_df_KB], axis=1)
        
        _, best_intrinsic, best_dist= find_df_min_value(score_df)
        best_model_proposal.append({'dist': best_dist, 'intrinsic': best_intrinsic})

    df['proposal_cam_model_predict'] = best_model_proposal
    df.to_excel(os.path.join(data_path, 'poposal_' + sampling_type + '_.xlsx'), index=False)