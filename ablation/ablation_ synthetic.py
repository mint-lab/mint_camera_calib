import os
import cv2 as cv
import glob
import numpy as np
from scipy.spatial.distance import hamming


def make_clean_directory(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    else:
        all_files = glob.glob(os.path.join(path_folder, '*.[jJpP]*[gG]'))
        for file in all_files:
            os.remove(file)

def multi_resolution_score(descriptor, resolution_level):
    return sum(descriptor) * (2**resolution_level)

def multi_resolution_descriptor(img, chessboard_pattern, resolution_level=None, cell_size=None):
    corners = find_chessboard_corners(img, chessboard_pattern) 
    if corners is None:
        raise ValueError("Can not find corners")
    
    img_w, img_h = img.shape[1], img.shape[0]
    if resolution_level is not None:
        cell_width = img_w//(2**resolution_level)
        cell_height = img_h//(2**resolution_level)
    elif cell_size is not None:
        cell_height = cell_size
        cell_width = cell_size

    descriptor = np.zeros(len(range(0, img_w - 1, cell_width)) * len(range(0, img_h - 1, cell_height)))
    ind = 0
    for i in range(0, img_h - 1, cell_height): 
        for j in range(0, img_w - 1, cell_width):
            top_left_pt = (j, i)
            bottom_right_pt = (min(j + cell_width, img_w - 1), min(i + cell_height, img_h - 1))
            for conner in corners:
                if top_left_pt [0] < conner[0][0] < bottom_right_pt[0] and top_left_pt[1] < conner[0][1] < bottom_right_pt[1]:
                    descriptor[ind] = 1 
            ind += 1

    return descriptor

def resolution_map(img, chessboard_pattern, resolution_level=None, cell_size=None):
    img = img.copy()
    corners= find_chessboard_corners(img, chessboard_pattern)
    if corners is None:
        raise ValueError("Can not find corners")
    for corner in corners:
        x, y = corner[0].astype(int)
        img = cv.circle(img, (x, y), radius=6, color=(0, 255, 0), thickness=-100)
    img_w, img_h = img.shape[1], img.shape[0]
    
    if resolution_level is not None:
        cell_width = img_w//(2**resolution_level)
        cell_height = img_h//(2**resolution_level)
    elif cell_size is not None:
        cell_height = cell_size
        cell_width = cell_size
    else:
        raise ValueError('Assign value for resolution_level or cell_size!!!!!')

    for i in range(cell_width, img_w - 1, cell_width):
        start_pt = (i, 0)
        end_pt = (i, img_h - 1)
        img = cv.line(img, start_pt, end_pt, color=(0, 0, 255), thickness=1)

    for i in range(cell_height, img_h - 1, cell_height):
        start_pt = (0, i)
        end_pt = (img_w - 1, i)
        img = cv.line(img, start_pt, end_pt, color=(0, 0, 255), thickness=1)
    
    return img

def laplacian_blur_metric(img):
    # Convert img to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Compute gradient by Laplacian
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    
    # Compute bluriness
    blur_score = np.mean(laplacian ** 2)
    
    return blur_score

def find_chessboard_corners(images, board_pattern):
    # Find 2D corner points from given images
    gray = cv.cvtColor(images, cv.COLOR_BGR2GRAY)
    complete, pts = cv.findChessboardCorners(gray, board_pattern)

    corners = None
    if complete:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)

    return corners

def filter_img_select(data_dir, output_path, cell_size, resolution_levels, chessboard_pattern,
                      dis_thresh=None, blur_thresh=None, multi_score_thresh=None, check_mode=None):
    # Select images
    all_descriptors = []
    img_select = []
    make_clean_directory(output_path)

    imgs_dir = glob.glob(os.path.join(data_dir, '*.[jJpP]*[gG]'))
    imgs = [cv.imread(img) for img in imgs_dir]

    for img, img_path in zip(imgs, imgs_dir):
        if check_mode == 'wo.PD':
            print('='*50)
            print('image:', os.path.basename(img_path))
            print(laplacian_blur_metric(img))
            if laplacian_blur_metric(img) > blur_thresh:
                img_select.append(img)
                cv.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)   
        elif check_mode == 'wo.PQ':
            if len(img_select) == 0:
                    img_select.append(img)
                    cv.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)
                    all_descriptors.append(multi_resolution_descriptor(img, chessboard_pattern, cell_size=cell_size))
            else:
                new_frame_descriptor = multi_resolution_descriptor(img, chessboard_pattern, cell_size=cell_size)
                hamming_distances = [round(hamming(new_frame_descriptor, descriptor), 2) 
                                     for descriptor in all_descriptors]
                new_frame_multi_descriptors = [multi_resolution_descriptor(img, chessboard_pattern, 
                                                                           resolution_level=resolution_level) 
                                                                           for resolution_level in resolution_levels]
                multi_score = sum(multi_resolution_score(descriptor, resolution_level) 
                                  for descriptor, resolution_level in zip(new_frame_multi_descriptors, resolution_levels))
                
                if all(distance >= dis_thresh for distance in hamming_distances) and multi_score >= multi_score_thresh:
                    img_select.append(img)
                    cv.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)
                    all_descriptors.append(new_frame_descriptor)
                else:
                    print('='*50)
                    print('hamming_distances:', hamming_distances)
                    print('multi_score:', multi_score)
                    print('multi_resolution_score_threshold:', multi_score_thresh)
        else:
            if laplacian_blur_metric(img) > blur_thresh:
                if len(img_select) == 0:
                    img_select.append(img)
                    cv.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)
                    all_descriptors.append(multi_resolution_descriptor(img, chessboard_pattern, cell_size=cell_size))
                else:
                    new_frame_descriptor = multi_resolution_descriptor(img, chessboard_pattern, cell_size=cell_size)
                    hamming_distances = [round(hamming(new_frame_descriptor, descriptor), 2) for descriptor in all_descriptors]
                    new_frame_multi_descriptors = [multi_resolution_descriptor(img, chessboard_pattern, resolution_level=resolution_level) for resolution_level in resolution_levels]
                    multi_score = sum(multi_resolution_score(descriptor, resolution_level) for descriptor, resolution_level in zip(new_frame_multi_descriptors, resolution_levels))
                    if all(distance >= dis_thresh for distance in hamming_distances) and multi_score >= multi_score_thresh:
                        img_select.append(img)
                        cv.imwrite(os.path.join(output_path, os.path.basename(img_path)), img)
                        all_descriptors.append(new_frame_descriptor)

    return img_select

if __name__ == '__main__':
    imgs_dir = 'data/synthetic/ablation1'
    output_path = 'data/synthetic/'
    chessboard_pattern = (5, 4)
    dis_thresh = 0.5
    multi_score_thresh = 1.5
    resolution_levels = [1, 2, 3]
    cell_size = 32
    blur_thresh = 150
    
    # ablation_mode = 'wo.PD'
    # blur_thresh = 150

    # ablation_mode = 'wo.PQ'
    # dis_thresh = 0.01
    # multi_score_thresh = 100
    # resolution_levels = [1, 2, 3]
    # cell_size = 32

    ablation_mode = 'our'
    blur_thresh = 150
    dis_thresh = 0.01 
    multi_score_thresh = 100
    resolution_levels = [1, 2, 3]
    cell_size = 32

    output_path = os.path.join(output_path, ablation_mode)
    img_select = filter_img_select(imgs_dir, output_path, cell_size, resolution_levels, 
                                   chessboard_pattern, dis_thresh, blur_thresh, 
                                   multi_score_thresh, check_mode=ablation_mode)
    # for img in img_select:
        
    print('===========================img_select============================')
    # print(img_select)
    print(len(img_select))