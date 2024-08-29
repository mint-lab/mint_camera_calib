import numpy as np
import cv2 as cv
from cam_cali_select import CameraCalibration 

if __name__ == '__main__':
    data_path = 'data/real/ISAW'
    pattern = (10, 7)
    intrinsic_type = 'P4' 
    dist_type = 'KB2'

    camera_calibration = CameraCalibration(data_path)

    # Load images
    imgs, img_name = camera_calibration.load_img()
    if not imgs:
        raise ValueError("No images found in the specified path.")
    
    # Find chessboard corners
    img_pts, img_size = camera_calibration.find_chessboard_corners(imgs, pattern)
    if not img_pts:
        raise ValueError("Chessboard corners not found in any of the images.")
    
    # Generate object points
    obj_pts = camera_calibration.generate_obj_points(pattern, len(img_pts))

    # Make calibration flag
    flags = camera_calibration.make_cali_flag(intrinsic_type=intrinsic_type, dist_type=dist_type)

    # Calibrate camera
    rms, K, dist_coef, rvecs, tvecs = camera_calibration.calibrate(obj_pts, img_pts, img_size, dist_type=dist_type, flags=flags)

    # Read image for undistortion
    img_path = 'data/real/ISAW/frame_11.jpg'
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    h,  w = img.shape[:2]
    
    # Undistort image
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist_coef, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, K, dist_coef)
    print(dst)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # Display results
    cv.imshow('Distorted Image', img)
    cv.imshow('Undistorted Image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()