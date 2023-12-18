import numpy as np
import cv2 as cv
import glob
import pandas as pd
from tqdm import tqdm
import os
import random
from pprint import pprint

class ObjPoints:
    def __init__(self, board_pattern):
        self.board_pattern = board_pattern

    def initilize_obj_points(self, len_img_points):
        obj_pts = [[c, r, 0] for r in range(self.board_pattern[1]) for c in range(self.board_pattern[0])]
        obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
        row, col = obj_points[0].shape
        obj_points = [x.reshape(row, 1, col) for x in obj_points]
        return obj_points


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


class CalibrationFlag:
    def __init__(self):
        self.intrinsic            = {}
        self.intrinsic['normal']  = {}
        self.intrinsic['fisheye'] = {}

        self.intrinsic['normal']['f']            = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
        self.intrinsic['normal']['fx_fy']        = cv.CALIB_FIX_PRINCIPAL_POINT
        self.intrinsic['normal']['f_cx_cy']      =  cv.CALIB_FIX_ASPECT_RATIO

        self.intrinsic['fisheye']['f']           = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT + cv.fisheye.CALIB_FIX_FOCAL_LENGTH
        self.intrinsic['fisheye']['fx_fy']       = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT
        self.intrinsic['fisheye']['f_cx_cy']     = cv.fisheye.CALIB_FIX_FOCAL_LENGTH
    

        self.distortion            = {}
        self.distortion['normal']  = {}
        self.distortion['fisheye'] = {}
        
        self.distortion['normal']['k1']            = cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.distortion['normal']['k1_k2']         = cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.distortion['normal']['k1_k2_k3']      = cv. CALIB_ZERO_TANGENT_DIST
        self.distortion['normal']['no']            = cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        
        self.distortion['fisheye']['k1']           = cv.fisheye.CALIB_FIX_K2 + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4  
        self.distortion['fisheye']['k1_k2']        = cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        self.distortion['fisheye']['k1_k2_k3']     = cv.fisheye.CALIB_FIX_K4
        self.distortion['fisheye']['no']           = cv.fisheye.CALIB_FIX_K1 + cv.fisheye.CALIB_FIX_K2  + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4

    def make_cali_flag(self, intrinsic_type, dist_type, cam_type='normal'):
        fisheye_calib_flag = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        normal_calib_flag = None

        intrinsic_dict = self.intrinsic[cam_type]
        distortion_dict = self.distortion[cam_type]

        if cam_type == 'fisheye':
            if dist_type == 'k1_k2_k3_k4' and intrinsic_type == 'fx_fy_cx_cy':
                return fisheye_calib_flag
            elif intrinsic_type == 'fx_fy_cx_cy':
                return fisheye_calib_flag + distortion_dict[dist_type]
            elif dist_type == 'k1_k2_k3_k4':
                return fisheye_calib_flag + intrinsic_dict[intrinsic_type]
            else:
                return fisheye_calib_flag + intrinsic_dict[intrinsic_type] + distortion_dict[dist_type]
        else:
            if dist_type == 'k1_k2_p1_p2_k3' and intrinsic_type == 'fx_fy_cx_cy':
                return normal_calib_flag
            elif intrinsic_type == 'fx_fy_cx_cy':
                return distortion_dict[dist_type]
            elif dist_type == 'k1_k2_p1_p2_k3':
                return intrinsic_dict[intrinsic_type]
            else:
                return intrinsic_dict[intrinsic_type] + distortion_dict[dist_type]


class CameraModelCombination:
    def __init__(self):
        self.intrinsic               = {}
        self.intrinsic['normal']     = ['f', 'fx_fy', 'f_cx_cy', 'fx_fy_cx_cy']
        self.intrinsic['fisheye']    = ['f', 'fx_fy', 'f_cx_cy', 'fx_fy_cx_cy']
    

        self.distortion              = {}
        self.distortion['normal']    = ['no', 'k1', 'k1_k2', 'k1_k2_k3', 'k1_k2_p1_p2_k3']
        self.distortion['fisheye']   = ['no', 'k1', 'k1_k2', 'k1_k2_k3', 'k1_k2_k3_k4']


def load_img(input_path):
    files = glob.glob(input_path + '*.png')
    img_select = [cv.imread(file) for file in files]
    return img_select


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


def calibrate(train_obj_points, train_img_points, img_size, cam_type='normal', K=None, dist=None, flag_cali=None):
    if cam_type == 'fisheye':
        return cv.fisheye.calibrate(train_obj_points, train_img_points, img_size[::-1], K=K, D=dist, flags=flag_cali)
    else:
        return cv.calibrateCamera(train_obj_points, train_img_points, img_size[::-1], cameraMatrix=K, distCoeffs=dist, flags=flag_cali)


def find_reproject_points(obj_points, rvecs, tvecs, K, dist, cam_type='normal'):
    if cam_type == 'fisheye':
        reproj_img_points, _ = cv.fisheye.projectPoints(obj_points, rvecs, tvecs, K, dist)
    else:
        reproj_img_points, _ = cv.projectPoints(obj_points, rvecs, tvecs, K, dist)

    return reproj_img_points


def cal_error(obj_points, img_points, rvecs, tvecs, K, dist, cam_type='normal'):
    mean_error = 0
    for i in range(len(obj_points)):
        reproj_img_points = find_reproject_points(obj_points[i], rvecs[i], tvecs[i], K, dist, cam_type)
        error = cv.norm(img_points[i], reproj_img_points, cv.NORM_L2) / np.sqrt(len(reproj_img_points))
        mean_error += error

    return mean_error / len(obj_points)


def train_test_process(obj_points_train, img_points_train, obj_points_test, img_points_test, img_size, f, dist, cam_type='normal'):
    # Create a flag for calibration
    cali_flag = CalibrationFlag()
    flags_calibrate = cali_flag.make_cali_flag(focal_length=f, distortion=dist, cam_type=cam_type)

    # TRAIN
    rms_train, K_train, dist_train, rvecs_train, tvecs_train = calibrate(obj_points_train, img_points_train, 
                                                                        img_size, flag_cali=flags_calibrate, cam_type=cam_type)
    # TEST
    train_error = cal_error(obj_points_train, img_points_train, rvecs_train, tvecs_train, K_train, dist_train, cam_type=cam_type)
    test_error = cal_error(obj_points_test, img_points_test, rvecs_train, tvecs_train, K_train, dist_train, cam_type=cam_type)

    return train_error, test_error


def caculate_model_score(train_error, test_error, f, dist):
    test_error_weight = 1
    train_error_weight = 1
    train_test_weight = 0.05
    num_parameter_weight = 0.05

    if dist == 'no':
        num_para = len(f.split('_'))
    else:
        num_para = len(f.split('_')) + len(dist.split('_'))

    train_test_ratio = test_error / train_error

    return test_error_weight * test_error + train_error_weight * train_error + train_test_weight * train_test_ratio + num_parameter_weight * num_para


if __name__ == '__main__':
    num_data = 1
    input_file = 'data/image_' + str(num_data) + '/'
    result_path = 'results/'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    chessboard_pattern = (10, 7)
    images = load_img(input_file)
    img_points, img_size = find_chessboard_corners(images, chessboard_pattern)
  
    # Create object points data
    obj_3d = ObjPoints(chessboard_pattern)
    obj_points= obj_3d.initilize_obj_points(len(img_points))

    cam_types = ['normal', 'fisheye']
    data_sampling_types = ['extrapolar', 'interpolar', 'random']
    cam_model = CameraModelCombination()

    best_combination = {}
    model_best_score = 100
    combination_results = []
    for cam_type in tqdm(cam_types):
        frame = []
        focal_lengths = cam_model.focal_length[cam_type]
        distortions = cam_model.distortion[cam_type]
        for data_sampling_type in data_sampling_types:
            # Split data into train and test data
            data_maker = DataSampling(obj_points, img_points, chessboard_pattern)
            obj_points_train, img_points_train, obj_points_test, img_points_test = data_maker.split_data(sampling_type=data_sampling_type)

            results = []
            for f in focal_lengths:
                result = []
                for dist in distortions:
                    train_error, test_error = train_test_process(obj_points_train, img_points_train, obj_points_test, img_points_test, img_size, f, dist, cam_type)
                    model_score = caculate_model_score(train_error, test_error, f, dist)
                    if model_score < model_best_score:
                        model_best_score = model_score
                        best_combination['cam_type'] = cam_type
                        best_combination['focal_lengths'] = f
                        best_combination['distortions'] = dist
                        best_combination['train_error'] = train_error
                        best_combination['test_error'] = test_error
                        best_combination['model_best_score'] = model_best_score
                    result.append({'train': f"{round(train_error, 2):.2f}", 'test': f"{round(test_error, 2):.2f}"})
                results.append(result)   

            df = pd.DataFrame(results, index= focal_lengths, columns=distortions)
            frame.append(df)
        combination_results.append(frame)
    pprint(best_combination)
               
    with pd.ExcelWriter(result_path + 'results_' + str(num_data) + '.xlsx') as writer:
        for df_ind, df in enumerate(combination_results):
            df = pd.concat(df, axis=0)
            df.to_excel(writer, sheet_name='all', startcol=df_ind * (df.shape[1] + 1))
            df.to_excel(writer, sheet_name= cam_types[df_ind], index=False)