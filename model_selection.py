import matplotlib.pyplot as plt
import seaborn as sns
from camera_calibration import *

def load_img(img_pts_path):
    all_files = glob.glob(os.path.join(img_pts_path, '*.png'))
    img_pts = [cv.imread(file) for file in all_files]
    return img_pts

def find_chessboard_corners(images, board_pattern):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
    assert len(img_points) > 0

    return img_points, gray.shape

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

def vizualize_score(df):
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=0.65) # for label size
    sns.heatmap(df, annot=True, cmap='jet', square=True) # font size
    # plt.axis('equal')
    plt.show()

def normalize_df(df):
    return (df - df.stack().min()) / (df.stack().max() - df.stack().min())

if __name__ == '__main__':
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    data_path = 'data/real/data_40/img'
    data_type = 'real'
    proj_model_BC = ['P0', 'P1', 'P2', 'P3']
    dist_model_BC = ['BC0', 'BC1', 'BC2', 'BC3']
    proj_model_KB = ['P1', 'P3']
    dist_model_KB =['KB0', 'KB1', 'KB2']

    if data_type == 'real':
        imgs = load_img(data_path)
        img_pts, img_size = find_chessboard_corners(imgs, chessboard_pattern)
    elif data_type == 'synthetic':
        img_pts = load_img_pts(data_path)

    obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
    N_samples = len(obj_pts) * obj_pts[0].shape[0]

    RMSE_BC = cali_error_process(obj_pts, img_pts, proj_model_BC, dist_model_BC, img_size)
    RMSE_KB = cali_error_process(obj_pts, img_pts, proj_model_KB, dist_model_KB, img_size)

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
        
    BIC_BC = cal_BIC(RMSE_BC, N_samples, num_para_BC)
    BIC_KB = cal_BIC(RMSE_KB, N_samples, num_para_KB)
    BIC_df_BC = pd.DataFrame(BIC_BC, index=proj_model_BC, columns=dist_model_BC)
    BIC_df_KB = pd.DataFrame(BIC_KB, index=proj_model_KB, columns=dist_model_KB)
    BIC_df = pd.concat([BIC_df_BC, BIC_df_KB], axis=1)
    _, min_BIC_intrinsic, min_BIC_dist= find_df_min_value(BIC_df)
    
    print('================================== RMSE ==================================')
    print(RMSE_df, '\n')
    print('================================== BIC ===================================')
    print(BIC_df, '\n')
    print('============================ BSET MODEL BIC ==============================')
    print(normalize_df(BIC_df))
    print('Projection Model = ', min_BIC_intrinsic)
    print('Distortion Model = ', min_BIC_dist)

    vizualize_score (normalize_df(BIC_df))
