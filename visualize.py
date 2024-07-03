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

    return array

def visualize_model_wise(df):
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=0.65) # for label size
    sns.heatmap(df, annot=True, cmap='jet', square=True) # font size
    # plt.axis('equal')
    plt.show()

def visualize_point_wise(error, img_name):
    plt.imshow(error, cmap='jet', interpolation='nearest', aspect='auto')
    plt.colorbar()  
    plt.title('Point wise error')  
    plt.xlabel('Image') 
    plt.xticks(ticks=np.arange(0, len(img_name)), labels=img_name, rotation='vertical')
    plt.ylabel('Point')
    plt.tight_layout()
    plt.show()

def normalize_df(df):
    return (df - df.stack().min()) / (df.stack().max() - df.stack().min())


if __name__ == '__main__':
    # Add arguments thi the parser
    parser = argparse.ArgumentParser(prog='visualize', description='Visualization of model wise score or point wise rmse')
    parser.add_argument('result_file', type=str, help='specify the result file path')
    parser.add_argument('-t', '--type_visualization', default='model_wise_socre', type=str, help='specify if user want to display model wise score or point wise rmse')
    parser.add_argument('-p', '--chessboard_pattern', default=(10, 7), type=tuple, help='Specify the chessboard pattern size')

    # Parse the command-line arguments
    args = parser.parse_args()

    file_path = args.result_file
    with open(file_path, 'r') as json_file:
        results = json.load(json_file)

    if args.type_visualization == 'model_wise_socre':
        model_wise_df = results['model_wise_score']
        model_wise_df = pd.DataFrame.from_dict(model_wise_df, orient='index')
        visualize_model_wise(normalize_df(model_wise_df))
    elif args.type_visualization == 'point_wise_rmse':
        pattern = args.chessboard_pattern
        imgs, img_name = CameraCalibration.load_img(results['data_path'])
        img_pts, img_size = CameraCalibration.find_chessboard_corners(imgs, pattern)

        obj_pts = CameraCalibration.generate_obj_points(pattern, len(img_pts))
        N_samples = len(obj_pts) * obj_pts[0].shape[0]
        # cam_w, cam_h = img_size[1], img_size[0]

        flags = CameraCalibration.make_flag(intrinsic_type = results['best_proj_model'], dist_type = results['best_dist_model'])
        rms, K, dist_coef, rvecs, tvecs = CameraCalibration.calibrate(obj_pts, img_pts, img_size, dist_type = results['best_dist_model'], flags=flags)
        pt_wwise_error = []
        for i in range(len(obj_pts)):
            reproj_img_pts = CameraCalibration.find_reproject_points(obj_pts[i], rvecs[i], tvecs[i], K, dist_coef, dist_type = results['best_dist_model'])
            diff = reproj_img_pts - img_pts[i]
            pt_wwise_error.append(np.array([cv.norm(d, cv.NORM_L2) for d in diff]))
        
        pt_wwise_error = np.array(pt_wwise_error).T
        pt = [f'pt{i}' for i in range(1, pattern[0] * pattern[1] + 1)]
        point_wise_rmse = pd.DataFrame(pt_wwise_error, columns=img_name, index=pt)
        file_name = [file.split('.')[0] for file in img_name]
        visualize_point_wise(pt_wwise_error, file_name)
        
        