import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import hamming
import glob
import random


def make_clean_directory(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    else:
        all_files = glob.glob(os.path.join(path_folder, '*.[jJpP]*[gG]'))
        for file in all_files:
            os.remove(file)

def filter_img_select(video_file, output_path, cell_size, resolution_levels, 
                      chessboard_pattern, dis_thresh, blur_thresh, 
                      multi_score_thresh, check_mode=None):
    # Open a video
    wnd_name = 'Image Selection'
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    all_descriptors = []
    img_select = []
    ind = 1
    
    make_clean_directory(output_path)

    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        # Show the image
        cv.putText(img, f'Number Selected Image: {ind-1}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))
        cv.imshow(wnd_name, img)

        # Process the key event 
        key = cv.waitKey(10)
        if key == ord(' '):             # Space: Pause and show corners
            key = cv.waitKey()
            if key == ord('\r'):             
                if check_mode == 'PQ':
                    print('='*50)
                    print(laplacian_blur_metric(img))
                    if laplacian_blur_metric(img) > blur_thresh:
                        img_select.append(img)
                        cv.imwrite(os.path.join(output_path, f'img_{ind}.jpg'), img)
                        ind += 1  
                elif check_mode == 'PD':
                    if len(img_select) == 0:
                            img_select.append(img)
                            cv.imwrite(os.path.join(output_path, f'img_{ind}.jpg'), img)
                            all_descriptors.append(multi_resolution_descriptor(img, chessboard_pattern, cell_size=cell_size))
                            ind += 1
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
                            cv.imwrite(os.path.join(output_path, f'img_{ind}.jpg'), img)
                            all_descriptors.append(new_frame_descriptor)
                            ind += 1
                        # else:
                        print('='*50)
                        print('hamming_distances:', hamming_distances)
                        print('multi_score:', multi_score)
                        print('multi_resolution_score_threshold:', multi_score_thresh)
                else:
                    if laplacian_blur_metric(img) > blur_thresh:
                        if len(img_select) == 0:
                            img_select.append(img)
                            cv.imwrite(os.path.join(output_path, f'img_{ind}.jpg'), img)
                            all_descriptors.append(multi_resolution_descriptor(img, chessboard_pattern, cell_size=cell_size))
                            ind += 1
                        else:
                            new_frame_descriptor = multi_resolution_descriptor(img, chessboard_pattern, cell_size=cell_size)
                            hamming_distances = [round(hamming(new_frame_descriptor, descriptor), 2) for descriptor in all_descriptors]
                            new_frame_multi_descriptors = [multi_resolution_descriptor(img, chessboard_pattern, resolution_level=resolution_level) for resolution_level in resolution_levels]
                            multi_score = sum(multi_resolution_score(descriptor, resolution_level) for descriptor, resolution_level in zip(new_frame_multi_descriptors, resolution_levels))
                            if all(distance >= dis_thresh for distance in hamming_distances) and multi_score >= multi_score_thresh:
                                img_select.append(img)
                                cv.imwrite(os.path.join(output_path, f'img_{ind}.jpg'), img)
                                all_descriptors.append(new_frame_descriptor)
                                ind += 1
                            
        if key == 27:                  # ESC: Exit (Complete image selection)
            break

    cv.destroyAllWindows()
    return img_select

def random_img_select(video_file, num_select_frame, output_path, seed=1000):
     # Open the video file
    video_capture = cv.VideoCapture(video_file)
    
    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # If the frame was read successfully, add it to the list
        if ret:
            frames.append(frame)
        else:
            # Break the loop if there are no more frames to read
            break
   
    # Release the video capture object
    video_capture.release()

    if seed is not None:
        random.seed(seed)
    
    # Ensure we do not try to select more frames than are available
    if num_select_frame > len(frames):
        print("Error: Not enough frames to select from.")
        return []

    # Randomly select `num_frames` from the list of frames
    random_frames = random.sample(frames, num_select_frame)

    save_folder = os.path.join(output_path, 'random')
    print('save_folder = ', save_folder)
    make_clean_directory(save_folder)
    save_frames(random_frames, save_folder)

    return random_frames

def save_frames(frames, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each frame as an image file
    for i, frame in enumerate(frames):
        file_path = os.path.join(output_folder, f'frame_{i + 1}.jpg')  # Save as JPG
        cv.imwrite(file_path, frame)

def find_chessboard_corners(images, board_pattern):
    # Find 2D corner points from given images
    gray = cv.cvtColor(images, cv.COLOR_BGR2GRAY)
    complete, pts = cv.findChessboardCorners(gray, board_pattern)

    corners = None
    if complete:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)

    return corners

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
 

if __name__ == '__main__':
    video_file = 'data/video/chessboard.avi' 
    select_type = 'filter'  
    check_mode = 'PD' 
    img_save_folder = 'data/real/cross_validation_30/PD'
    chessboard_pattern = (10, 7) 
    cell_width = 30
    resolution_levels = [2, 3, 4] 
    multi_score_thresh = 1000
    dis_thresh = 0.12 
    blur_thresh = 450

    if select_type == 'random':
        img_select = random_img_select(video_file, output_path=img_save_folder, num_select_frame=40, seed=100)
    elif select_type == 'filter':
        img_select = filter_img_select(video_file, output_path=img_save_folder, cell_size=cell_width, 
                                       chessboard_pattern=chessboard_pattern, resolution_levels=resolution_levels, 
                                       dis_thresh=dis_thresh, blur_thresh=blur_thresh, multi_score_thresh=multi_score_thresh,
                                       check_mode=check_mode)