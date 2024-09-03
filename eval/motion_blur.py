import cv2 as cv
import numpy as np
import os
import glob
from scipy.spatial.distance import hamming


def motion_blur(image, kernel_size, type):
    # Tạo ma trận kernel cho motion blur
    kernel = np.zeros((kernel_size, kernel_size))
    if type ==  'vertical':
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    elif type =='horizontal':
        kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size
    
    # Áp dụng convolution với kernel đã tạo
    blurred = cv.filter2D(image, -1, kernel)
    return blurred

def blur_score(image):
    # Chuyển đổi ảnh sang ảnh grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Sử dụng bộ lọc Sobel để tính gradient theo x và y
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    
    # Tính tổng bình phương của gradient theo x và y
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Tính độ mờ bằng cách lấy giá trị trung bình của gradient
    blur_score = np.mean(gradient_magnitude)
    
    return blur_score

def laplacian_blur_metric(image):
    # Chuyển đổi ảnh sang ảnh grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Tính toán gradient bằng Laplacian
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    
    # Tính toán độ mờ bằng cách lấy giá trị trung bình bình phương của gradient
    blur_score = np.mean(laplacian ** 2)
    
    return blur_score

def resolution_map(img, cell_width):
    img_w, img_h = img.shape[1], img.shape[0]

    for i in range(cell_width, img_w - 1, cell_width):
        start_pt = (i, 0)
        end_pt = (i, img_h - 1)
        img = cv.line(img, start_pt, end_pt, color=(0, 0, 255), thickness=1)

    for i in range(cell_width, img_h - 1, cell_width):
        start_pt = (0, i)
        end_pt = (img_w, i)
        img = cv.line(img, start_pt, end_pt, color=(0, 0, 255), thickness=1)

    cv.imshow('anh', img)  
    cv.waitKey(0)
    cv.DestroyWindow()     

def find_chessboard_corners(images, board_pattern):
    # Find 2D corner points from given images
    gray = cv.cvtColor(images, cv.COLOR_BGR2GRAY)
    _, pts = cv.findChessboardCorners(gray, board_pattern)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)

    return corners

def create_descriptor(img, cell_width, chessboard_pattern):
    img_w, img_h = img.shape[1], img.shape[0]
    conners = find_chessboard_corners(img, chessboard_pattern) 

    descriptor = np.zeros(len(range(0, img_w - 1, cell_width)) * len(range(0, img_h - 1, cell_width)))
    ind = 0
    for i in range(0, img_h - 1, cell_width): 
        for j in range(0, img_w - 1, cell_width):
            top_left_pt = (j, i)
            bottom_right_pt = (min(j + cell_width, img_w - 1), min(i + cell_width, img_h - 1))
            for conner in conners:
                if top_left_pt [0] < conner[0][0] < bottom_right_pt[0] and top_left_pt[1] < conner[0][1] < bottom_right_pt[1]:
                    descriptor[ind] = 1 
            ind += 1
    return descriptor

def random_img_select(video_file, img_path, cell_width, chessboard_pattern, select_all=False, wait_msec=10, wnd_name='Image Selection'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    ind = 0
    filter_img_select = []
    ind_filter = 1
    
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    else:
        all_files = glob.glob(os.path.join(img_path, '*.[jJpP]*[gG]'))
        for file in all_files:
            os.remove(file)

    if not os.path.exists('random_filter'):
        os.makedirs('random_filter')
    else:
        all_files = glob.glob(os.path.join('random_filter', '*.[jJpP]*[gG]'))
        for file in all_files:
            os.remove(file)

    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            cv.putText(img, f'NSelect: {ind}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, img)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):             # Space: Pause and show corners
                key = cv.waitKey()
                if key == ord('\r'):
                    save_path = os.path.join(img_path , 'img_' + str(ind + 1) +'.jpg')
                    cv.imwrite(save_path, img)
                    cv.imshow(wnd_name, img)
                    img_select.append(img) # Enter: Select the image
                    ind += 1
                     
                    if len(filter_img_select) == 0:
                        save_path = os.path.join('random_filter' , 'img_' + str(ind_filter) +'.jpg')
                        cv.imwrite(save_path, img)
                        filter_img_select.append(img)
                        ind_filter += 1
                    else:
                        d1 = create_descriptor(img, cell_width, chessboard_pattern)
                        d2 = create_descriptor(filter_img_select[-1], cell_width, chessboard_pattern)
                        d = hamming(d1, d2) 
                        print('d = ', d) 
                        if d > 0.1:
                            save_path = os.path.join('random_filter' , 'img_' + str(ind_filter) +'.jpg')
                            cv.imwrite(save_path, img)
                            filter_img_select.append(img)
                            ind_filter += 1

            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select