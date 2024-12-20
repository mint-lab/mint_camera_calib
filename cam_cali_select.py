import json
import os
import argparse
import numpy as np
import cv2 as cv
import glob
import pandas as pd


class CameraCalibration:
    def __init__(self, img_path, config_file='cfgs/config.json') -> None:
        self.img_path = img_path
        self.config_file = config_file
        self.config = self.get_default_config()

        # Project model for BC
        self.proj_model_BC          = {}
        self.proj_model_BC['P1']    = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P2']    = cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P3']    = cv.CALIB_FIX_ASPECT_RATIO

        # Projection model for KB
        self.proj_model_KB          = {}
        self.proj_model_KB['P2']    = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT
    
        # Distortion model
        self.dist_model             = {}
        self.dist_model['BC0']      = cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC1']      = cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC2']      = cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC4']      = cv.CALIB_FIX_K3
        self.dist_model['KB0']      = cv.fisheye.CALIB_FIX_K1 + cv.fisheye.CALIB_FIX_K2  + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        self.dist_model['KB1']      = cv.fisheye.CALIB_FIX_K2 + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4  
        self.dist_model['KB2']      = cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        
        # Try to load the configuration user defined
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.config.update(config)
        except FileNotFoundError:
            print('Model selection will use the default configuration')
    
    @staticmethod
    def get_default_config():
        config = {}

        config['proj_model_BC'] = ['P1', 'P2', 'P3', 'P4']
        config['dist_model_BC'] = ['BC0', 'BC1', 'BC2', 'BC4']
        config['proj_model_KB'] = ['P2', 'P4']
        config['dist_model_KB'] = ['KB0', 'KB1', 'KB2']
        return config
    
    def generate_obj_pts(self, board_pattern, img_pts):

        obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
        obj_pts = [np.array(obj_pts, dtype=np.float32)] * len(img_pts)  # Must be 'np.float32'
        row, col = obj_pts[0].shape

        return [x.reshape(row, 1, col) for x in obj_pts]
    
    def load_img(self):
        # All image formats
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.jp2']

        # Filter all image files
        all_files = [os.path.join(self.img_path, file) for file in os.listdir(self.img_path)
                    if os.path.splitext(file)[1].lower() in image_extensions]
        sorted_files = sorted(all_files)
        # Read
        imgs = []
        img_name = []
        imgs = [cv.imread(file) for file in sorted_files]
        img_name = [os.path.basename(file) for file in sorted_files]

        return imgs, img_name
    
    def load_img_pts(self):
        all_files = glob.glob(os.path.join(self.img_path, 'img*.npy'))
        sorted_files = sorted(all_files)
        img_pts = [np.load(file) for file in sorted_files]
        return img_pts
    
    def find_chessboard_corners(self, imgs, board_pattern):
        """
            Finds the 2D corner points of a chessboard pattern in the provided images.
            
            Args:
                imgs (list): List of images in which to find the chessboard corners.
                board_pattern (tuple): Number of internal corners per chessboard row and column.
                
            Returns:
                tuple: A tuple containing the list of found image points and the image shape.
                
            Raises:
            ValueError: If no chessboard corners are found in any of the images.
        """
        img_points = []
        for img in imgs:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            complete, pts = cv.findChessboardCorners(gray, board_pattern)
            if complete:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
                img_points.append(corners)
        if not img_points:
            raise ValueError("No chessboard corners found in any of the images.")

        return img_points, gray.shape

    def make_cali_flag(self, intrinsic_type, dist_type):
        KB_flag = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW

        if dist_type.startswith('BC'):
            if intrinsic_type == 'P4':
                return self.dist_model[dist_type]
            else:
                return self.proj_model_BC[intrinsic_type] + self.dist_model[dist_type]
        else:
            if intrinsic_type == 'P4':
                return KB_flag + self.dist_model[dist_type]
            else:
                return KB_flag + self.proj_model_KB[intrinsic_type] + self.dist_model[dist_type]

    def calibrate(self, obj_pts, img_pts, img_size, dist_type, flags, K=None, dist_coef=None):
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
    
    def cal_rmse_for_models(self, obj_pts, img_pts, proj_model, dist_model, img_size):
        RMSEs = []
        for intrinsic_type in proj_model:
            RMSE = []
            for dist_type in dist_model:
                flags = self.make_cali_flag(intrinsic_type, dist_type)
                rms, K, dist_coef, rvecs, tvecs= self.calibrate(obj_pts, img_pts, img_size, dist_type=dist_type, flags=flags)
                RMSE.append(rms)
            RMSEs.append(RMSE)
        return np.array(RMSEs)
    
    def find_reproject_points(self, obj_pts, rvecs, tvecs, K, dist_coef, dist_type):
        if dist_type.startswith('KB'):
            reproj_img_points, _ = cv.fisheye.projectPoints(obj_pts, rvecs, tvecs, K, dist_coef)
        else:
            reproj_img_points, _ = cv.projectPoints(obj_pts, rvecs, tvecs, K, dist_coef)

        return reproj_img_points
    
    def model_wise_rmse(self):
        imgs, img_name = self.load_img()
        img_pts, img_size = self.find_chessboard_corners(imgs, self.config['chessboard_pattern'])

        obj_pts = self.generate_obj_pts(self.config['chessboard_pattern'], img_pts)
        num_pts_in_dataset = len(obj_pts) * obj_pts[0].shape[0]

        # Calibration
        RMSE_BC = self.cal_rmse_for_models(obj_pts, img_pts, self.config['proj_model_BC'], self.config['dist_model_BC'], img_size)
        RMSE_KB = self.cal_rmse_for_models(obj_pts, img_pts, self.config['proj_model_KB'], self.config['dist_model_KB'], img_size)

        RMSE_BC_df = pd.DataFrame(RMSE_BC, index=self.config['proj_model_BC'], columns=self.config['dist_model_BC'])
        RMSE_KB_df = pd.DataFrame(RMSE_KB, index=self.config['proj_model_KB'], columns=self.config['dist_model_KB'])
        RMSE_df = pd.concat([RMSE_BC_df, RMSE_KB_df], axis=1)

        return RMSE_df, num_pts_in_dataset, img_name


class CameraSelection:
    def __init__(self, img_path, save_path, config_file='cfgs/config.json') -> None:
        self.img_path = img_path
        self.save_path = save_path
        self.config_file = config_file
        self.config = self.get_default_config()

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.config.update(config)
        except FileNotFoundError:
            print('Model selection will use the default configuration')

    @staticmethod
    def get_default_config():
        config = {}

        config['proj_num_para'] = {}
        config['proj_num_para']['P1'] = 1
        config['proj_num_para']['P2'] = 2
        config['proj_num_para']['P3'] = 3
        config['proj_num_para']['P4'] = 4

        config['dist_num_para'] = {}
        config['dist_num_para']['BC0'] = 0
        config['dist_num_para']['BC1'] = 1
        config['dist_num_para']['BC2'] = 2
        config['dist_num_para']['BC4'] = 4
        config['dist_num_para']['KB0'] = 0
        config['dist_num_para']['KB1'] = 1
        config['dist_num_para']['KB2'] = 2

        config['proj_model_BC'] = ['P1', 'P2', 'P3', 'P4']
        config['dist_model_BC'] = ['BC0', 'BC1', 'BC2', 'BC4']
        config['proj_model_KB'] = ['P2', 'P4']
        config['dist_model_KB'] = ['KB0', 'KB1', 'KB2']

        return config

    def apply_criteria(self, RMSE, N_samples, dist_model, proj_model):
        num_para = self.config['proj_num_para'][proj_model] + self.config['dist_num_para'][dist_model] 
        if self.config['criteria'] == 'AIC':
            return N_samples * np.log(pow(RMSE, 2)) + 2 * num_para
        elif self.config['criteria'] =='BIC':
            return N_samples * np.log(pow(RMSE, 2)) + num_para * np.log10(N_samples)
     
    def score_models(self, RMSE_df, N_samples):  
        score_df = RMSE_df.copy()

        for proj_model in RMSE_df.index:
            for dist_model in RMSE_df.columns:
                RMSE = RMSE_df.at[proj_model, dist_model]
                score_df.at[proj_model,dist_model] = self.apply_criteria(RMSE, N_samples, dist_model, proj_model)

        return score_df

    def find_df_min_value(self, df):
        min_value = df.stack().min()
        min_index, min_column = df.stack().idxmin()

        camera_calibration = CameraCalibration(self.img_path)
        imgs, img_name = camera_calibration.load_img()
        img_pts, img_size = camera_calibration.find_chessboard_corners(imgs, self.config['chessboard_pattern'])
        obj_pts = camera_calibration.generate_obj_pts(self.config['chessboard_pattern'], img_pts)

        flags = camera_calibration.make_cali_flag(intrinsic_type=min_index, dist_type=min_column)
        rms, K, dist_coef, rvecs, tvecs = cam_model_calibration.calibrate(obj_pts, img_pts, img_size,  dist_type=min_column, flags=flags)

        return min_value, rms, min_index, min_column, K, dist_coef, rvecs, tvecs

    def run_selection(self, RMSE_df, num_pts_in_dataset):
        score_df = self.score_models(RMSE_df, num_pts_in_dataset)
        min_score, rms, score_min_intrinsic, score_min_dist, K, dist_coef, rvecs, tvecs = self.find_df_min_value(score_df)       

        print('================================== Model wise RMSE=====================================')
        print(RMSE_df, '\n')
        print('================================== Model wise Score ===================================')
        print(score_df, '\n')
        print(f"============================ BEST MODEL {self.config['criteria']} ==============================")
        print('Projection Model = ', score_min_intrinsic)
        print('Distortion Model = ', score_min_dist)
        print('RMSE selected model = ', rms)
        print('Best score = ', min_score)

        results = {
            'data_path': self.img_path,
            'score_best_model':{'proj_model': score_min_intrinsic, 'dist_model': score_min_dist},
            'rms': rms,
            'min_score': min_score,
            'model_wise_rms': RMSE_df.to_dict(orient='index'),
            'model_wise_score': score_df.to_dict(orient='index'), 
            'img_name': img_name, 
            'K': K.tolist(),
            'dist_coef': dist_coef.tolist(),
            'rvecs': [rvec.tolist() for rvec in rvecs],
            'tvecs': [tvec.tolist() for tvec in tvecs]
        }
        
        with open(self.save_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='model selection', description='Camera model calibration and selection')
    parser.add_argument('img_file', type=str, help='specify the image file path')
    parser.add_argument('save_file', type=str, help='specify the file path to save the result')
    parser.add_argument('-c', '--config_file', default='cfgs/cam_cali_select.json', type=str, help='specify a configuration file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Camera calibration
    cam_model_calibration = CameraCalibration(args.img_file, args.config_file)
    RMSE_df, num_pts_in_dataset, img_name = cam_model_calibration.model_wise_rmse()

    # Camera model selection
    camera_model_selection = CameraSelection(args.img_file, args.save_file, args.config_file)

    # min_rms, rms_min_intrinsic, rms_min_dist = camera_model_selection.run_selection(RMSE_df)
    camera_model_selection.run_selection(RMSE_df, num_pts_in_dataset)