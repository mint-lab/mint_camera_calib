import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_img(img_path):
    # All image formats
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.jp2']

    # Filter all image files
    all_files = [os.path.join(img_path, file) for file in os.listdir(img_path)
                 if os.path.splitext(file)[1].lower() in image_extensions]

    # Read
    imgs = [cv.imread(file) for file in all_files]

    return imgs

def find_chessboard_corners(imgs, board_pattern, obj_pt):
    # Find 2D corner points from given images
    img_points = []
    obj_pts = []
    for img in imgs:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            obj_pts.append(obj_pt)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
    assert len(img_points) > 0

    return img_points, obj_pts, gray.shape

def generate_obj_points(board_pattern, len_img_points):
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32)] * len_img_points # Must be 'np.float32'
    row, col = obj_points[0].shape
    obj_points = [x.reshape(row, 1, col) for x in obj_points]
    return obj_points

class CalibrationFlag:
    def __init__(self):
        self.proj_model_BC = {}
        self.proj_model_BC['P0'] = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P1'] = cv.CALIB_FIX_PRINCIPAL_POINT
        self.proj_model_BC['P2'] = cv.CALIB_FIX_ASPECT_RATIO

        self.proj_model_KB = {}
        self.proj_model_KB['P1'] = cv.fisheye.CALIB_FIX_PRINCIPAL_POINT

        self.dist_model = {}
        self.dist_model['BC0'] = cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC1'] = cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC2'] = cv.CALIB_FIX_K3 + cv.CALIB_ZERO_TANGENT_DIST
        self.dist_model['BC3'] = cv.CALIB_FIX_K3
        self.dist_model['KB0'] = cv.fisheye.CALIB_FIX_K1 + cv.fisheye.CALIB_FIX_K2 + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        self.dist_model['KB1'] = cv.fisheye.CALIB_FIX_K2 + cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4
        self.dist_model['KB2'] = cv.fisheye.CALIB_FIX_K3 + cv.fisheye.CALIB_FIX_K4

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

if __name__ == '__main__':
    img_path = 'data/real/cross_validation_30/random'
    pattern = (10, 7)

    imgs = load_img(img_path)
    obj_pt = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj_pt[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    # obj_pt[:, 2] -= 10 

    img_pts, obj_pts, img_size = find_chessboard_corners(imgs, pattern, obj_pt)
    obj_pts = generate_obj_points(pattern, len(img_pts))
    dist_type1 = 'BC3'
    intrinsic_type1 = 'P3'
    calibrate_flag = CalibrationFlag()
    flags = calibrate_flag.make_flag(intrinsic_type1, dist_type1)
    rms, K, dist_coef, rvecs, tvecs = calibrate(obj_pts, img_pts, img_size, dist_type=dist_type1, flags=flags)

    # Print the calibration results
    print("Camera Matrix:\n", K)
    print("Distortion Coefficients:\n", dist_coef)

    # Visualize the camera poses in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw chessboard points
    ax.scatter(obj_pt[:, 0], obj_pt[:, 1], obj_pt[:, 2], s=50, c='r', marker='+')

    # Draw the camera positions
    for rvec, tvec in zip(rvecs, tvecs):
        # Convert rotation vector to rotation matrix
        R, _ = cv.Rodrigues(rvec)
        
        # Camera position in world coordinates
        camera_position = -R.T @ tvec
        

        # Draw a line from the camera center pointing along the z-axis of the camera
        # ax.scatter(camera_position[0], camera_position[1], camera_position[2], s=100, c='b', marker='^')
        # z_axis = R.T @ np.array([[0, 0, 1]]).T
        # ax.quiver(camera_position[0], camera_position[1], camera_position[2],
        #         z_axis[0], z_axis[1], z_axis[2], length=1, color='b')

        # Draw the camera's orientation (X, Y, Z axes)
        x_axis = R.T @ np.array([[1, 0, 0]]).T
        y_axis = R.T @ np.array([[0, 1, 0]]).T
        z_axis = R.T @ np.array([[0, 0, 1]]).T
        

        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                x_axis[0], x_axis[1], x_axis[2], color='r', length=1, label='X axis')
        
        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                y_axis[0], y_axis[1], y_axis[2], color='g', length=1, label='Y axis')
        
        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                z_axis[0], z_axis[1], z_axis[2], color='b', length=1, label='Z axis')
    
    # # Add coordinate frame at world origin
    # origin = np.array([0, 0, 0])
    # ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='black', length=1.5)
    # ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='black', length=1.5)
    # ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='black', length=1.5)
    # ax.text(0, 0, 0, 'Origin', color='black')

    # Label the axes
    ax.set_xlabel('X world')
    ax.set_ylabel('Y world')
    ax.set_zlabel('Z world')

    plt.show()