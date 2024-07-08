import os
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
from model_selection import CameraCalibration


def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    array = (array - min_val) / (max_val - min_val)

    return np.round(array, 4)

def visualize_model_wise(df):
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=0.65) # for label size
    sns.heatmap(df, annot=True, cmap='jet', square=True) # font size
    # plt.axis('equal')
    plt.show()

def visualize_point_wise(error, img_name):
    plt.imshow(error, cmap='Oranges', interpolation='nearest', aspect='auto')
    plt.colorbar()  
    plt.title('Point-wise RMSE')  
    plt.xlabel('Point Index') 
    plt.xticks(ticks=np.arange(0, error.shape[1], 5))
    plt.ylabel('Image')
    plt.yticks(ticks=np.arange(0, error.shape[0]), labels=img_name)
    plt.tight_layout()
    plt.show()

def normalize_df(df):
    return (df - df.stack().min()) / (df.stack().max() - df.stack().min())


if __name__ == '__main__':
    # Add arguments thi the parser
    parser = argparse.ArgumentParser(prog='visualize', description='Visualization of model wise score or point wise rmse')
    parser.add_argument('result_file', type=str, help='specify the result file path')
    parser.add_argument('-t', '--type_visualization', default='model_wise', type=str, help='specify if user want to display model wise score or point wise rmse')
    parser.add_argument('-p', '--chessboard_pattern', default=(10, 7), type=tuple, help='Specify the chessboard pattern size')

    # Parse the command-line arguments
    args = parser.parse_args()

    file_path = args.result_file
    with open(file_path, 'r') as json_file:
        results = json.load(json_file)

    if args.type_visualization == 'model_wise':
        model_wise_df = results['model_wise_score']
        model_wise_df = pd.DataFrame.from_dict(model_wise_df, orient='index')
        visualize_model_wise(normalize_df(model_wise_df))
    elif args.type_visualization == 'point_wise':
        pattern = args.chessboard_pattern
        camera_calibration = CameraCalibration(results['data_path'])
        imgs, img_name = camera_calibration.load_img()
        img_pts, img_size = camera_calibration.find_chessboard_corners(imgs, pattern)
        obj_pts = camera_calibration.generate_obj_points(pattern, len(img_pts))

        flags = camera_calibration.make_flag(intrinsic_type = results['best_proj_model'], dist_type = results['best_dist_model'])
        rms, K, dist_coef, rvecs, tvecs = camera_calibration.calibrate(obj_pts, img_pts, img_size, dist_type = results['best_dist_model'], flags=flags)
        pt_wwise_error = []
        for i in range(len(obj_pts)):
            reproj_img_pts = camera_calibration.find_reproject_points(obj_pts=obj_pts[i], rvecs=rvecs[i], tvecs=tvecs[i], K=K, dist_coef=dist_coef, dist_type=results['best_dist_model'])
            diff = reproj_img_pts - img_pts[i]
            pt_wwise_error.append(np.array([cv.norm(d, cv.NORM_L2) for d in diff]))
        
        pt_wise_error = np.array(pt_wwise_error)
        pts = [f'pt{i}' for i in range(1, pattern[0] * pattern[1] + 1)]
        point_wise_rmse = pd.DataFrame(pt_wise_error, columns=pts, index=img_name)
        file_name = [file.split('.')[0] for file in img_name]
        visualize_point_wise(pt_wise_error, file_name)