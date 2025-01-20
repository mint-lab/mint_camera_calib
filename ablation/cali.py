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
    def find_reproj_error(self, obj_points, img_points, rvecs, tvecs, K, dist_coef, dist_type):
        mean_error = 0
        for i in range(len(obj_points)):
            reproj_img_points = self.find_reproject_points(obj_points[i], rvecs[i], tvecs[i], K, dist_coef, dist_type)
            error = cv.norm(img_points[i], reproj_img_points, cv.NORM_L2) / np.sqrt(len(reproj_img_points))
            mean_error += error

        return mean_error / len(obj_points)

    def estimate_pose(self, pts_3d, pts_2d, K, distort, dist_type='BC'):
        '''Estimate the pose from 2D and 3D points with a fisheye camera'''
        rvecs = []
        tvecs = []
        for pt_3d, pt_2d in zip(pts_3d, pts_2d):
            if dist_type == 'BC':
                pt_2d_undistort = cv.undistortPoints(pt_2d, K, distort)
            else:
                pt_2d_undistort = cv.fisheye.undistortPoints(pt_2d.reshape(-1, 1, 2), K, distort)
            _, rvec, tvec = cv.solvePnP(pt_3d, pt_2d_undistort, np.eye(3), np.zeros(4))
            rvecs.append(rvec)
            tvecs.append(tvec)
        return rvecs, tvecs


if __name__ == '__main__':
    img_file = 'data/real/ISAW'
    config_file = 'cfgs/config.json'
    chess_pattern = (10, 7)

    # GT
    # fx_gt = 950.0
    # fy_gt = 1000.0
    # cx_gt = 639.5
    # cy_gt = 479.5
    # dist_coef_gt = np.array([0.05, 0.0, 0.0, 0.0, 0.0])

    # Camera calibration
    intrinsic_type = 'P4'
    dist_type = 'KB2' 
    cali = CameraCalibration(img_file, config_file)
    flags = cali.make_cali_flag(intrinsic_type, dist_type)

    imgs, img_name = cali.load_img()
    img_pts, img_size = cali.find_chessboard_corners(imgs, chess_pattern)
    obj_pts = cali.generate_obj_pts(chess_pattern, img_pts)

    rms, K, dist_coef, rvecs, tvecs= cali.calibrate(obj_pts, img_pts, img_size, dist_type=dist_type, flags=flags)
    print('K = ', K)
    print('rms = ', rms)
    # f_error = np.sqrt((fx_gt - K[0, 0])**2 + (fy_gt - K[1, 1])**2)
    # c_error = np.sqrt((cx_gt - K[0, 2])**2 + (cy_gt - K[1, 2])**2)
    # dist_coef = dist_coef.flatten()
    # dist_coef_gt = dist_coef_gt[:4] if dist_type.startswith('KB') else dist_coef_gt
    # dist_error = np.linalg.norm(dist_coef_gt - dist_coef)
    # print('f_error = ', f_error)
    # print('c_error = ', c_error)
    # print('dist_error = ', dist_error)

    # K = np.array([[1.3493e+03, 0.0000e+00, 6.4000e+02],
    #      [0.0000e+00, 1.3493e+03, 4.8000e+02],
    #      [0.0000e+00, 0.0000e+00, 1.0000e+00]])
    # dist_coef = np.array([0, 0.0, 0, 0])

    # # Estimate pose
    # rvec, tvec = cali.estimate_pose(obj_pts, img_pts, K, dist_coef, dist_type='KB')
    # error = cali.find_reproj_error(obj_pts, img_pts, rvec, tvec, K, dist_coef, dist_type)
    # print('='*20)
    # print('re_proj_error = ', error)
    