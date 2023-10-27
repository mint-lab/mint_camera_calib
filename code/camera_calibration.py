import numpy as np
import cv2 as cv
import glob


class ObjPoints:
    def __init__(self, board_pattern):
        self.board_pattern = board_pattern

    def initilize_ojb_points(self, len_img_points):
        obj_pts = [[c, r, 0] for r in range(self.board_pattern[1]) for c in range(self.board_pattern[0])]
        obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
        return obj_points


class MakeData:
    def __init__(self, obj_pts, img_pts, board_pattern):
        self.obj_pts = obj_pts
        self.img_pts = img_pts
        self.board_pattern = board_pattern

    def split_data(self, split_type):
        obj_points_train = []
        img_points_train = []
        obj_points_test = []
        img_points_test = []

        selected_slice_train = []
        selected_slice_test = []

        if split_type == 1:
            for i in range(self.board_pattern[1]):
                for j in range (self.board_pattern[0]):
                    if i == 0 or i == self.board_pattern[1] - 1:
                        selected_slice_test.append(i * self.board_pattern[1] + j)
                    else:
                        selected_slice_train.append(i * self.board_pattern[0] + j)


        obj_points_train = [i[np.ix_(selected_slice_train)] for i in self.obj_pts]
        img_points_train = [i[np.ix_(selected_slice_train)] for i in self.img_pts]
        obj_points_test = [i[np.ix_(selected_slice_test)] for i in self.obj_pts]
        img_points_test = [i[np.ix_(selected_slice_test)] for i in self.img_pts]

        return obj_points_train, img_points_train, obj_points_test, img_points_test


class CameraModel:
    focal_length = ['f', 'fx_fy', 'f_cx_cy', 'fx_fy_cx_cy']
    distortion = ['k1', 'k1_k2', 'k1_k2_k3', 'k1_k2_p1_p2_k3']


class CalbrationFlag:
    def __init__(self):
        self.focal_length = {}

        self.focal_length['f']       = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
        self.focal_length['fx_fy']   = cv.CALIB_FIX_PRINCIPAL_POINT
        self.focal_length['f_cx_cy'] =  cv.CALIB_FIX_ASPECT_RATIO
    

        self.distortion = {}
        
        self.distortion['k1']       = cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.distortion['k1_k2']    = cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        self.distortion['k1_k2_k3'] = cv. CALIB_ZERO_TANGENT_DIST
        self.distortion['no']       = cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3  + cv.CALIB_ZERO_TANGENT_DIST
        

    def make_cali_flag(self, focal_length, distortion):
        if distortion == 'k1_k2_p1_p2_k3' and focal_length == 'fx_fy_cx_cy':
            return None
        elif focal_length == 'fx_fy_cx_cy':
            return self.distortion[distortion]
        elif distortion == 'k1_k2_p1_p2_k3': 
            return self.focal_length[focal_length]
        else:
            return self.focal_length[focal_length] + self.distortion[distortion]


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
            corners = cv.cornerSubPix(gray, pts, (11,11), (-1,-1), criteria)
            img_points.append(corners)
    assert len(img_points) > 0  

    # Calibrate the camera
    return img_points, gray.shape


def calibrate(train_obj_points, train_img_points, img_size, K=None, dist=None, flag_cali=None):
    return cv.calibrateCamera(train_obj_points, train_img_points, img_size[::-1], cameraMatrix=K, distCoeffs=dist, flags=flag_cali)


def cal_error(obj_points, img_points, rvecs, tvecs, K, dist):
    mean_error = 0
    for i in range(len(obj_points)):
        reproj_img_points, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        error = cv.norm(img_points[i], reproj_img_points, cv.NORM_L2)/len(reproj_img_points)
        mean_error += error

    return mean_error/len(obj_points)


if __name__ == '__main__':
    input_file = 'data/image/'
    chessboard_pattern = (10, 7)

    images = load_img(input_file)
    img_points, img_size = find_chessboard_conners(images, chessboard_pattern)
  
    obj_3d = ObjPoints(chessboard_pattern)
    
    obj_points= obj_3d.initilize_ojb_points(len(img_points))
    data_maker = MakeData(obj_points, img_points, chessboard_pattern)
    train_obj_points, train_img_points, test_obj_points, test_img_points = data_maker.split_data(split_type=1)


    # TRAIN
    config_camera = CameraModel()
    best_choice_f = 0
    best_choice_dist = 0
    final_train_error = 0
    final_test_error = 0
    best_ratio = 100
    best_K = 0
    best_coeff = 0

    
    for f in config_camera.focal_length:
        for dist in config_camera.distortion:
            flag_cal = CalbrationFlag()
            flags_calibrate = flag_cal.make_cali_flag(focal_length=f, distortion=dist)
            rms_train, K_train, dist_train, rvecs_train, tvecs_train = calibrate(train_obj_points, train_img_points, 
                                                                                img_size, flag_cali=flags_calibrate)
            # TEST
            train_error = cal_error(train_obj_points, train_img_points, rvecs_train, tvecs_train, K_train, dist_train) 
            test_error = cal_error(test_obj_points, test_img_points, rvecs_train, tvecs_train, K_train, dist_train)
            ratio = train_error + test_error
   
        
        if ratio < best_ratio: 
            best_ratio = ratio
            best_choice_f = f
            best_choice_dist = dist
            final_train_error = train_error
            final_test_error = test_error
            best_K = K_train
            best_dist = dist_train

    print(best_ratio)
    print('focal length: ', best_choice_f)
    print('distortion: ', best_choice_dist)
    print('final_train_error: ', final_train_error)
    print('final_test_error: ', final_test_error)
    print('Distortion: ', best_dist)
    print('Camera matrix: ', K_train)