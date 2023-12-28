import glob
import os
import random
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm


def generate_obj_points(board_pattern, len_img_points):
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
    row, col = obj_points[0].shape
    obj_points = [x.reshape(row, 1, col) for x in obj_points]
    return obj_points

def load_img_pts(img_pts_path):
    all_files = glob.glob(os.path.join(img_pts_path, '*.npy'))
    img_pts = [np.load(file) for file in all_files]
    return img_pts

def calibrate(obj_pts, img_pts, img_size, dist_type, flags, K=None, dist_coef=None):
    if dist_type.startswith('BC'):
        return cv.calibrateCamera(obj_pts, img_pts, img_size[::-1], cameraMatrix=K, distCoeffs=dist_coef, flags=flags)
    else:
        try:
            return cv.fisheye.calibrate(obj_pts, img_pts, img_size[::-1], K=K, D=dist_coef, flags=flags)
        except:
            flags -= cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            print(dist_type)
            print(flags)
            return cv.fisheye.calibrate(obj_pts, img_pts, img_size[::-1], K=K, D=dist_coef, flags=flags)

'''
def cal_aic(RMSE, dist_type, intrinsic_type, N_samples):
    proj_model_num_para = {'P0':1, 'P1':2, 'P2': 3, 'P3': 4}
    dist_model_num_para = {'BC0': 0, 'BC1': 1, 'BC2': 2, 'BC3': 3, 'BC4': 5, 
                           'KB0': 0, 'KB1': 1, 'KB2': 2, 'KB3': 3, 'KB4': 4}
    
    num_para = proj_model_num_para[intrinsic_type] + dist_model_num_para[dist_type]

    return N_samples * np.log(pow(RMSE, 2)) + 2 * num_para

def cal_bic(RMSE, dist_type, intrinsic_type, N_samples):
    proj_model_num_para = {'P0':1, 'P1':2, 'P2': 3, 'P3': 4}
    dist_model_num_para = {'BC0': 0, 'BC1': 1, 'BC2': 2, 'BC3': 3, 'BC4': 5, 
                           'KB0': 0, 'KB1': 1, 'KB2': 2, 'KB3': 3, 'KB4': 4}
    
    num_para = proj_model_num_para[intrinsic_type] + dist_model_num_para[dist_type]

    return N_samples * np.log(pow(RMSE, 2)) + num_para * np.log10(N_samples)
'''

def cal_AIC(RMSE, N_samples):
    proj_model_num_para = [1, 2, 3, 4]
    dist_model_num_para = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4]

    num_para = np.array([[dist + f for dist in dist_model_num_para] for f in proj_model_num_para])

    return N_samples * np.log(pow(RMSE, 2)) + 2 * num_para

def cal_BIC(RMSE, N_samples):
    proj_model_num_para = [1, 2, 3, 4]
    dist_model_num_para = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4]

    num_para = np.array([[dist + f for dist in dist_model_num_para] for f in proj_model_num_para])

    return N_samples * np.log(pow(RMSE, 2)) + num_para * np.log10(N_samples)

def cali_error_process(obj_pts, img_pts, proj_model, dist_model, img_size):
    RMSEs = []
    for intrinsic_type in proj_model:
        RMSE = []
        for dist_type in dist_model:
            calibrate_flag = CalibrationFlag()
            flags = calibrate_flag.make_flag(intrinsic_type, dist_type)
            rms, K, dist_coef, rvecs, tvecs= calibrate(obj_pts, img_pts, img_size, dist_type=dist_type, flags=flags)
            RMSE.append(rms)
        RMSEs.append(RMSE)
    return np.array(RMSEs)

def find_reproject_points(obj_points, rvecs, tvecs, K, dist_coef, dist_type):
    if dist_type.startswith('KB'):
        reproj_img_points, _ = cv.fisheye.projectPoints(obj_points, rvecs, tvecs, K, dist_coef)
    else:
        reproj_img_points, _ = cv.projectPoints(obj_points, rvecs, tvecs, K, dist_coef)

    return reproj_img_points

def find_reproj_error(obj_points, img_points, rvecs, tvecs, K, dist_coef, dist_type):
    mean_error = 0
    for i in range(len(obj_points)):
        reproj_img_points = find_reproject_points(obj_points[i], rvecs[i], tvecs[i], K, dist_coef, dist_type)
        error = cv.norm(img_points[i], reproj_img_points, cv.NORM_L2) / np.sqrt(len(reproj_img_points))
        mean_error += error

    return mean_error / len(obj_points)

def train_test_error_process(obj_pts_train, img_pts_train, obj_pts_test, img_pts_test, proj_model, dist_model, img_size):
    train_error_all = []
    test_error_all = []
    for intrinsic_type in proj_model:
        train_error = []
        test_error = []
        for dist_type in dist_model:
            calibrate_flag = CalibrationFlag()
            flags = calibrate_flag.make_flag(intrinsic_type, dist_type)
            rms, K, dist_coef, rvecs, tvecs= calibrate(obj_pts_train, img_pts_train, img_size, dist_type=dist_type, flags=flags)
            train_error.append(rms)

            rms_test = find_reproj_error(obj_pts_test, img_pts_test, rvecs, tvecs, K, dist_coef, dist_type)
            test_error.append(rms_test)

        train_error_all.append(train_error)
        test_error_all.append(test_error)
    return np.array(train_error_all), np.array(test_error_all)

def caculate_model_score(train_error, test_error):
    test_error_weight = 1e-3
    train_error_weight = 1e-3
    ratio_weight = 1

    train_test_ratio = train_error / test_error

    # return -np.log(test_error_weight / test_error + train_error_weight / train_error + ratio_weight * train_test_ratio)
    num_para = get_num_para()
    return np.log(train_error + test_error + pow(train_error, 2) / test_error + num_para * train_error)

def get_num_para():
    dists = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4]
    intrinsics = [1, 2, 3, 4]

    num_para = []
    for intrinsic in intrinsics:
        tmp = []
        for dist in dists:
            tmp.append(dist + intrinsic)
        num_para.append(tmp)

    return np.array(num_para)

def find_df_min_value(df):
    min_value = df.stack().min()
    min_index, min_column = df.stack().idxmin()

    return min_value, min_index, min_column

def add_noise(x, mean=0, standard_deviation=0.01):
    noise = np.random.normal(mean, standard_deviation, x.shape)
    x = x + noise
    x = np.float32(np.round(x, decimals = 4))

    return x

class CalibrationFlag:
    def __init__(self):
        self.proj_model_BC          = {}
        self.proj_model_BC['P0']    = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P1']    = cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P2']    =  cv.CALIB_FIX_ASPECT_RATIO

        self.proj_model_KB          = {}
        self.proj_model_KB['P0']    = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT + cv.fisheye.CALIB_FIX_FOCAL_LENGTH
        self.proj_model_KB['P1']    = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_KB['P2']    = cv.fisheye.CALIB_FIX_FOCAL_LENGTH
    
        self.dist_model             = {}
        self.dist_model['BC0']      = cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC1']      = cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC2']      = cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC3']      = cv. CALIB_ZERO_TANGENT_DIST
        self.dist_model['KB0']      = cv.fisheye.CALIB_FIX_K1 + cv.fisheye.CALIB_FIX_K2  + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        self.dist_model['KB1']      = cv.fisheye.CALIB_FIX_K2 + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4  
        self.dist_model['KB2']      = cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        self.dist_model['KB3']      = cv.fisheye.CALIB_FIX_K4

    def make_flag(self, intrinsic_type, dist_type):
        KB_flag = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        BC_flag = None

        if dist_type.startswith('BC'):
            if dist_type == 'BC4' and intrinsic_type == 'P3':
                return BC_flag
            elif intrinsic_type == 'P3':
                return self.dist_model[dist_type]
            elif dist_type == 'BC4':
                return self.proj_model_BC[intrinsic_type]
            else:
                return self.proj_model_BC[intrinsic_type] + self.dist_model[dist_type] 
        else:
            if dist_type == 'KB4' and intrinsic_type == 'P3':
                return KB_flag
            elif intrinsic_type == 'P3':
                return KB_flag + self.dist_model[dist_type]
            elif dist_type == 'KB4':
                return KB_flag + self.proj_model_KB[intrinsic_type]
            else:
                return KB_flag + self.proj_model_KB[intrinsic_type] + self.dist_model[dist_type]

class DataSampling:
    def __init__(self, obj_pts, img_pts, board_pattern):
        self.obj_pts = obj_pts
        self.img_pts = img_pts
        self.board_pattern = board_pattern

    def split_data(self, sampling_type, test_size=0.3):
        selected_slice_train, selected_slice_test = [], []
        
        if sampling_type == 'random':
            selected_slice_train, selected_slice_test = self.random_sampling(test_size)
        else:
            selected_slice_train, selected_slice_test = self.structure_sampling(sampling_type)

        obj_points_train = [i[selected_slice_train] for i in self.obj_pts]
        img_points_train = [i[selected_slice_train] for i in self.img_pts]
        obj_points_test = [i[selected_slice_test] for i in self.obj_pts]
        img_points_test = [i[selected_slice_test] for i in self.img_pts]

        return obj_points_train, img_points_train, obj_points_test, img_points_test

    def structure_sampling(self, sampling_type):
        points_in_length = []
        for i in range(self.board_pattern[1]):
            for j in range(self.board_pattern[0]):
                if i == 0 or i == self.board_pattern[1] - 1:
                    points_in_length.append(i * self.board_pattern[0] + j)

        points_in_breadth = []
        for i in range(self.board_pattern[0] * self.board_pattern[1]):
            if i % self.board_pattern[0] == 0 or (i + 1) % self.board_pattern[0] == 0:
                points_in_breadth.append(i)

        all_points = list(range(self.board_pattern[0] * self.board_pattern[1]))
        if sampling_type == 'extrapolar':
            selected_slice_test = list(set(points_in_length + points_in_breadth))
            selected_slice_train = [i for i in all_points if i not in selected_slice_test]  
        elif sampling_type == 'interpolar':
            selected_slice_train = list(set(points_in_length + points_in_breadth))
            selected_slice_test = [i for i in all_points if i not in selected_slice_train]

        return selected_slice_train, selected_slice_test

    def random_sampling(self, test_size):
        all_points = list(range(self.board_pattern[0] * self.board_pattern[1]))

        # Calculate the number of points to select (30% of the list size)
        num_points_to_select = int(test_size * len(all_points))

        # Randomly select 30% of elements
        random.seed(1000)
        selected_slice_test = sorted(random.sample(all_points, num_points_to_select))
        selected_slice_train = [i for i in all_points if i not in selected_slice_test]

        return selected_slice_train, selected_slice_test


if __name__ == '__main__':
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    save_path = 'data/synthetic/dataset'

    df = pd.read_excel(os.path.join(save_path, 'synthetic_data.xlsx'))
    img_pts_paths = df['path']
    cam_model = df['cam_model']
    proj_model = ['P0', 'P1', 'P2', 'P3']
    dist_model = ['BC0', 'BC1', 'BC2', 'BC3', 'BC4', 'KB0', 'KB1', 'KB2', 'KB3', 'KB4']

    best_model_AIC = []
    best_model_BIC = []
    best_model_proposal = []
    ind = 1
    for img_pts_path in tqdm(img_pts_paths):
        img_pts = load_img_pts(img_pts_path )
        obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
        N_samples = len(obj_pts) * obj_pts[0].shape[0]
        RMSE = cali_error_process(obj_pts, img_pts, proj_model, dist_model, img_size)

        AIC = cal_AIC(RMSE, N_samples)
        AIC_df = pd.DataFrame(AIC, index=proj_model, columns=dist_model)
        min_AIC, min_AIC_index, min_AIC_column = find_df_min_value(AIC_df)
        best_model_AIC.append({'dist': min_AIC_column, 'intrinsic': min_AIC_index})

        BIC = cal_BIC(RMSE, N_samples)
        BIC_df = pd.DataFrame(BIC, index=proj_model, columns=dist_model)
        min_BIC, min_BIC_index, min_BIC_column = find_df_min_value(BIC_df)
        best_model_BIC.append({'dist': min_BIC_column, 'intrinsic': min_BIC_index})

        data = DataSampling(obj_pts, img_pts, chessboard_pattern)
        sampling_type = 'extrapolar'
        obj_pts_train, img_pts_train, obj_pts_test, img_pts_test = data.split_data(sampling_type=sampling_type)
        train_error, test_error = train_test_error_process(obj_pts_train, img_pts_train, obj_pts_test,
                                                           img_pts_test, proj_model, dist_model, img_size)
        # np.save(os.path.join(save_path, 'train_error.npy'), train_error)
        # np.save(os.path.join(save_path, 'test_error.npy'), test_error)
        model_score = caculate_model_score(train_error, test_error)
        score_df = pd.DataFrame(model_score, index=proj_model, columns=dist_model)
        min_score, min_score_index, min_score_colums = find_df_min_value(score_df)
        best_model_proposal.append({'dist': min_score_colums, 'intrinsic': min_score_index})
        print('index = ', ind)
        ind += 1
        print('=========================')

    df['AIC_cam_model_predict'] = best_model_AIC
    df['BIC_cam_model_predict'] = best_model_BIC
    df['proposal_cam_model_predict'] = best_model_proposal
    df.to_excel(os.path.join(save_path, 'accuracy.xlsx'), index=False)