import glob
import pandas as pd
import numpy as np
import cv2 as cv

def generate_obj_points(board_pattern, len_img_points):
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
    row, col = obj_points[0].shape
    obj_points = [x.reshape(row, 1, col) for x in obj_points]
    return obj_points

def load_img_pts(img_pts_path):
    all_files = glob.glob(img_pts_path + '/' + '*.npy')
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

def AIC_process(obj_pts, img_pts, proj_model, dist_model, img_size):
    N_samples = len(obj_pts) * obj_pts[0].shape[0]
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
           

if __name__ == '__main__':
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    save_path = 'data/synthetic/dataset/'

    df = pd.read_excel(save_path + 'synthetic_data.xlsx')
    img_pts_paths = df['path']
    cam_model = df['cam_model']
    proj_model = ['P0', 'P1', 'P2', 'P3']
    dist_model = ['BC0', 'BC1', 'BC2', 'BC3', 'BC4', 'KB0', 'KB1', 'KB2', 'KB3', 'KB4']

    for img_pts_path in [img_pts_paths[0]]:
        # print(img_pts_path)
        img_pts = load_img_pts(img_pts_path + '/')
        obj_pts = generate_obj_points(chessboard_pattern, len(img_pts))
        AIC_df = AIC_process(obj_pts, img_pts, proj_model, dist_model, img_size)
        # print(AIC_df)