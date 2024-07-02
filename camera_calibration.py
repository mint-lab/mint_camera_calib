import glob
import os
import random
import numpy as np
import cv2 as cv


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

def generate_obj_points(board_pattern, len_img_points):
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
    row, col = obj_points[0].shape
    obj_points = [x.reshape(row, 1, col) for x in obj_points]
    return obj_points

def load_img_pts(img_pts_path):
    all_files = glob.glob(os.path.join(img_pts_path, 'img*.npy'))
    img_pts = [np.load(file) for file in all_files]
    return img_pts

def load_img(img_pts_path):
    # All image formats
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.jp2']

    # Filter all image files
    all_files = [os.path.join(img_pts_path, file) for file in os.listdir(img_pts_path)
                 if os.path.splitext(file)[1].lower() in image_extensions]

    # Readd
    img_pts = [cv.imread(file) for file in all_files]
    img_name = [os.path.basename(file) for file in all_files]

    return img_pts, img_name

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

def model_wise_rmse(obj_pts, img_pts, proj_model, dist_model, img_size):
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
    train_error_all, test_error_all = [], []
    for intrinsic_type in proj_model:
        train_error, test_error = [], []
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

def find_df_min_value(df):
    min_value = df.stack().min()
    min_index, min_column = df.stack().idxmin()

    return min_value, min_index, min_column

class CalibrationFlag:
    def __init__(self):
        self.proj_model_BC          = {}
        self.proj_model_BC['P0']    = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P1']    = cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P2']    = cv.CALIB_FIX_ASPECT_RATIO

        self.proj_model_KB          = {}
        self.proj_model_KB['P1']    = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT
    
        self.dist_model             = {}
        self.dist_model['BC0']      = cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC1']      = cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC2']      = cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC3']      = cv.CALIB_FIX_K3
        self.dist_model['KB0']      = cv.fisheye.CALIB_FIX_K1 + cv.fisheye.CALIB_FIX_K2  + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        self.dist_model['KB1']      = cv.fisheye.CALIB_FIX_K2 + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4  
        self.dist_model['KB2']      = cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4

    def make_flag(self, intrinsic_type, dist_type):
        KB_flag = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW

        if dist_type.startswith('BC'):
            if intrinsic_type == 'P3':
                return self.dist_model[dist_type]
            else:
                return self.proj_model_BC[intrinsic_type] + self.dist_model[dist_type] 
        else:
            if intrinsic_type == 'P3':
                return KB_flag + self.dist_model[dist_type]
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

        selected_slice_test, selected_slice_train = [], []
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