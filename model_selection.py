import json
import pandas as pd
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

def cal_score(RMSE, N_samples, num_para, criteria):
    if criteria == 'AIC':
        return N_samples * np.log(pow(RMSE, 2)) + 2 * num_para
    elif criteria =='BIC':
        return N_samples * np.log(pow(RMSE, 2)) + num_para * np.log10(N_samples)


def find_best_model(RMSE_BC, RMSE_KB, proj_model_BC, proj_model_KB, dist_model_BC, dist_model_KB, N_samples, criteria_type):
    num_para_BC = get_num_para_BC()
    num_para_KB = get_num_para_KB()

    BC = cal_score(RMSE_BC, N_samples, num_para_BC, criteria_type)
    KB = cal_score(RMSE_KB, N_samples, num_para_KB, criteria_type)
    df_BC = pd.DataFrame(BC, index=proj_model_BC, columns=dist_model_BC)
    df_KB = pd.DataFrame(KB, index=proj_model_KB, columns=dist_model_KB)
    df = pd.concat([df_BC, df_KB], axis=1)
    _, min_intrinsic, min_dist= find_df_min_value(df)

    return min_intrinsic, min_dist


if __name__ == '__main__':
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    data_path = 'data/quy'
    save_path = 'results/results.json'
    data_type = 'real'
    proj_model_BC = ['P0', 'P1', 'P2', 'P3']
    dist_model_BC = ['BC0', 'BC1', 'BC2', 'BC3']
    proj_model_KB = ['P1', 'P3']
    dist_model_KB = ['KB0', 'KB1', 'KB2']

    if data_type == 'real':
        imgs, img_name = load_img(data_path)
        img_pts, img_size = find_chessboard_corners(imgs, chessboard_pattern)
    elif data_type == 'synthetic':
        img_pts = load_img_pts(data_path)

    obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
    N_samples = len(obj_pts) * obj_pts[0].shape[0]

    RMSE_BC = model_wise_rmse(obj_pts, img_pts, proj_model_BC, dist_model_BC, img_size)
    RMSE_KB = model_wise_rmse(obj_pts, img_pts, proj_model_KB, dist_model_KB, img_size)

    RMSE_BC_df = pd.DataFrame(RMSE_BC, index=proj_model_BC, columns=dist_model_BC)
    RMSE_KB_df = pd.DataFrame(RMSE_KB, index=proj_model_KB, columns=dist_model_KB)
    RMSE_df = pd.concat([RMSE_BC_df, RMSE_KB_df], axis=1)

    num_para_BC = get_num_para_BC()
    num_para_KB = get_num_para_KB()

    # AIC_BC = cal_AIC(RMSE_BC, N_samples, num_para_BC)
    # AIC_KB = cal_AIC(RMSE_KB, N_samples, num_para_KB)
    # AIC_df_BC = pd.DataFrame(AIC_BC, index=proj_model_BC, columns=dist_model_BC)
    # AIC_df_KB = pd.DataFrame(AIC_KB, index=proj_model_KB, columns=dist_model_KB)
    # AIC_df = pd.concat([AIC_df_BC, AIC_df_KB], axis=1)
    # _, min_AIC_intrinsic, min_AIC_dist= find_df_min_value(AIC_df)
        
    BIC_BC = cal_score(RMSE_BC, N_samples, num_para_BC, 'BIC')
    BIC_KB = cal_score(RMSE_KB, N_samples, num_para_KB, 'BIC')
    BIC_df_BC = pd.DataFrame(BIC_BC, index=proj_model_BC, columns=dist_model_BC)
    BIC_df_KB = pd.DataFrame(BIC_KB, index=proj_model_KB, columns=dist_model_KB)
    BIC_df = pd.concat([BIC_df_BC, BIC_df_KB], axis=1)
    rmse_min, min_BIC_intrinsic, min_BIC_dist= find_df_min_value(BIC_df)
    
    print('================================== RMSE ==================================')
    print(RMSE_df, '\n')
    print('================================== BIC ===================================')
    print(BIC_df, '\n')
    print('============================ BEST MODEL BIC ==============================')
    print('Projection Model = ', min_BIC_intrinsic)
    print('Distortion Model = ', min_BIC_dist)

    # results = {
    #     'data_path': data_path,
    #     'best_proj_model': min_BIC_intrinsic,
    #     'best_dist_model': min_BIC_dist,
    #     'rmse_min': rmse_min,
    #     'model_wise_rmse': RMSE_df.to_dict(orient='index'),
    #     'model_wise_score': BIC_df.to_dict(orient='index')
    # }

    
    # with open(save_path, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)
