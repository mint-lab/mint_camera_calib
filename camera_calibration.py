import numpy as np
import cv2 as cv
import glob
import pandas as pd


class ObjPoints:
    def __init__(self, board_pattern):
        self.board_pattern = board_pattern

    def initilize_ojb_points(self, len_img_points):
        obj_pts = [[c, r, 0] for r in range(self.board_pattern[1]) for c in range(self.board_pattern[0])]
        obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
        row, col = obj_points[0].shape
        obj_points = [x.reshape(row, 1, col) for x in obj_points]
        return obj_points


class MakeTrainTestData:
    def __init__(self, obj_pts, img_pts, board_pattern):
        self.obj_pts = obj_pts
        self.img_pts = img_pts
        self.board_pattern = board_pattern

    def split_data(self, split_type):
        selected_slice_train, selected_slice_test = [], []
        
        if split_type == 1:
            selected_slice_train, selected_slice_test = self.split_type_1()
        elif split_type == 2:
            selected_slice_train, selected_slice_test = self.split_type_2()
        elif split_type == 3:
            selected_slice_train, selected_slice_test = self.split_type_3()
        elif split_type == 4:
            selected_slice_train, selected_slice_test = self.split_type_4()
        elif split_type == 5:
            selected_slice_train, selected_slice_test = self.split_type_5()

        obj_points_train = [i[np.ix_(selected_slice_train)] for i in self.obj_pts]
        img_points_train = [i[np.ix_(selected_slice_train)] for i in self.img_pts]
        obj_points_test = [i[np.ix_(selected_slice_test)] for i in self.obj_pts]
        img_points_test = [i[np.ix_(selected_slice_test)] for i in self.img_pts]

        return obj_points_train, img_points_train, obj_points_test, img_points_test

    def split_type_1(self):
        '''
        ***************************
        ---------------------------
        ---------------------------
        ---------------------------
        ***************************
        '''
        selected_slice_train, selected_slice_test = [], []
        for i in range(self.board_pattern[1]):
            for j in range(self.board_pattern[0]):
                if i == 0 or i == self.board_pattern[1] - 1:
                    selected_slice_test.append(i * self.board_pattern[0] + j)
                else:
                    selected_slice_train.append(i * self.board_pattern[0] + j)
        return selected_slice_train, selected_slice_test

    def split_type_2(self):
        '''
        *-------------------------*
        *-------------------------*
        *-------------------------*
        *-------------------------*
        *-------------------------*
        '''
        selected_slice_train, selected_slice_test = [], []
        for i in range(self.board_pattern[0] * self.board_pattern[1]):
            if i % self.board_pattern[0] == 0 or (i + 1) % self.board_pattern[0] == 0:
                selected_slice_test.append(i)
            else:
                selected_slice_train.append(i)
        return selected_slice_train, selected_slice_test

    def split_type_3(self):
        '''
        *-*-*-*-*-*-*-*-*-*-*-*-*
        -*-*-*-*-*-*-*-*-*-*-*-*-
        *-*-*-*-*-*-*-*-*-*-*-*-*
        -*-*-*-*-*-*-*-*-*-*-*-*-
        *-*-*-*-*-*-*-*-*-*-*-*-*
        '''
        selected_slice_train, selected_slice_test = [], []
        for i in range(self.board_pattern[0] * self.board_pattern[1]):
            if i % 2 == 0:
                selected_slice_train.append(i)
            else:
                selected_slice_test.append(i)
        return selected_slice_train, selected_slice_test
    
    def split_type_4(self):
        '''
        -*-*-*-*-*-*-*-*-*-*-*-*-
        -*-*-*-*-*-*-*-*-*-*-*-*-
        -*-*-*-*-*-*-*-*-*-*-*-*-
        -*-*-*-*-*-*-*-*-*-*-*-*-
        -*-*-*-*-*-*-*-*-*-*-*-*-
        '''
        selected_slice_train, selected_slice_test = [], []
        for i in range(self.board_pattern[1]):
            for j in range(self.board_pattern[0]):
                if j % 2 == 0:
                    selected_slice_train.append(i * self.board_pattern[0] + j)
                else:
                    selected_slice_test.append(i * self.board_pattern[0] + j)
        return selected_slice_train, selected_slice_test
    
    def split_type_5(self):
        '''
        ---------------------------
        ***************************
        ---------------------------
        ***************************
        ---------------------------
        '''
        selected_slice_train, selected_slice_test = [], []
        for i in range(self.board_pattern[1]):
            for j in range(self.board_pattern[0]):
                if i % 2 == 0:
                    selected_slice_train.append(i * self.board_pattern[0] + j)
                else:
                    selected_slice_test.append(i * self.board_pattern[0] + j)
        return selected_slice_train, selected_slice_test
    

class CalibrationFlag:
    def __init__(self):
        self.focal_length            = {}
        self.focal_length['normal']  = {}
        self.focal_length['fisheye'] = {}

        self.focal_length['normal']['f']            = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
        self.focal_length['normal']['fx_fy']        = cv.CALIB_FIX_PRINCIPAL_POINT
        self.focal_length['normal']['f_cx_cy']      =  cv.CALIB_FIX_ASPECT_RATIO

        self.focal_length['fisheye']['fx_fy']       = cv.fisheye.CALIB_FIX_SKEW + cv.fisheye.CALIB_FIX_PRINCIPAL_POINT
        self.focal_length['fisheye']['fx_fy_s']     = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT
        self.focal_length['fisheye']['fx_fy_cx_cy'] = cv.fisheye.CALIB_FIX_SKEW
    

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

    def make_cali_flag(self, focal_length, distortion, cam_type='normal'):

        if cam_type == 'fisheye':
            if distortion == 'k1_k2_k3_k4' and focal_length == 'fx_fy_cx_cy_s':
                return None
            elif focal_length == 'fx_fy_cx_cy_s':
                return self.distortion[cam_type][distortion]
            elif distortion == 'k1_k2_k3_k4': 
                return self.focal_length[cam_type][focal_length]
            else:
                return self.focal_length[cam_type][focal_length] + self.distortion[cam_type][distortion]
        else:   
            if distortion == 'k1_k2_p1_p2_k3' and focal_length == 'fx_fy_cx_cy':
                return None
            elif focal_length == 'fx_fy_cx_cy':
                return self.distortion[cam_type][distortion]
            elif distortion == 'k1_k2_p1_p2_k3': 
                return self.focal_length[cam_type][focal_length]
            else:
                return self.focal_length[cam_type][focal_length] + self.distortion[cam_type][distortion]


class CameraModel:
    def __init__(self):
        self.focal_length            = {}
        self.focal_length['normal']  = ['f', 'fx_fy', 'f_cx_cy', 'fx_fy_cx_cy']
        self.focal_length['fisheye'] = ['fx_fy', 'fx_fy_s', 'fx_fy_cx_cy', 'fx_fy_cx_cy_s']
    

        self.distortion              = {}
        self.distortion['normal']    = ['k1', 'k1_k2', 'k1_k2_k3', 'k1_k2_p1_p2_k3', 'no']
        self.distortion['fisheye']   = ['k1', 'k1_k2', 'k1_k2_k3', 'k1_k2_k3_k4', 'no']


def load_img(input_path):
    img_select = []
    files = glob.glob(input_path + '*.png')
    for file in files:
        img = cv.imread(file)
        img_select.append(img)
    return img_select


def find_chessboard_conners(images, board_pattern):

    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            criteria = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001
            corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
    assert len(img_points) > 0  

    return img_points, gray.shape


def calibrate(train_obj_points, train_img_points, img_size, cam_type='normal', K=None, dist=None, flag_cali=None):
    if cam_type == 'fisheye':
        return cv.fisheye.calibrate(train_obj_points, train_img_points, img_size[::-1], K=K, D=dist, flags=flag_cali)
    else:
        return cv.calibrateCamera(train_obj_points, train_img_points, img_size[::-1], cameraMatrix=K, distCoeffs=dist, flags=flag_cali)


def cal_error(obj_points, img_points, rvecs, tvecs, K, dist, cam_type='normal'):
    mean_error = 0
    for i in range(len(obj_points)):
        if cam_type == 'fisheye':
            reproj_img_points, _ = cv.fisheye.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        else:
            reproj_img_points, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        error = cv.norm(img_points[i], reproj_img_points, cv.NORM_L2) / len(reproj_img_points)
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


if __name__ == '__main__':
    input_file = 'data/image/'
    chessboard_pattern = (10, 7)

    images = load_img(input_file)
    img_points, img_size = find_chessboard_conners(images, chessboard_pattern)
  
    # Create object points data
    obj_3d = ObjPoints(chessboard_pattern)   
    obj_points= obj_3d.initilize_ojb_points(len(img_points))
    
    cam_types = ['fisheye', 'normal']
    data_split_types = [1, 2, 3, 4, 5]
    cam_model = CameraModel()

    for cam_type in cam_types:
        frame = []
        focal_lengths = cam_model.focal_length[cam_type]
        distortions =cam_model.distortion[cam_type]
        for data_split_type in data_split_types:
            # Split data into train and test data
            data = MakeTrainTestData(obj_points, img_points, chessboard_pattern)
            obj_points_train, img_points_train, obj_points_test, img_points_test = data.split_data(split_type=data_split_type)

            results = []
            for f in focal_lengths:
                result = []
                for dist in distortions:
                    train_error, test_error = train_test_process(obj_points_train, img_points_train, obj_points_test, img_points_test, img_size, f, dist, cam_type)
                    result.append({'train': round(train_error, 3), 'test': round(test_error, 3)})
                results.append(result)   
            
        
            df = pd.DataFrame(results, index= focal_lengths, columns=distortions)
            frame.append(df)

        with pd.ExcelWriter(cam_type + '_results.xlsx') as writer:
            df = pd.concat(frame, axis=0)
            df.to_excel(writer, sheet_name='all')
            for ind, df in enumerate(frame):
                df.to_excel(writer, sheet_name= 'data_type = ' + str(ind + 1), index=False)