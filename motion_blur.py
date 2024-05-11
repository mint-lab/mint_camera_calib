import cv2 as cv
import numpy as np


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