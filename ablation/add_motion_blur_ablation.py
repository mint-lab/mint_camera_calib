import numpy as np
import cv2 as cv
import os
import glob
import random
import matplotlib.pyplot as plt

def motion_blur(image, kernel_size, type):
    kernel = np.zeros((kernel_size, kernel_size))
    if type == 'vertical':
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    elif type == 'horizontal':
        kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred = cv.filter2D(image, -1, kernel)
    return blurred

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    return rotated

def translate_image(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))
    return translated

def scale_image(image, scale):
    (h, w) = image.shape[:2]
    scaled = cv.resize(image, (int(w * scale), int(h * scale)))
    return scaled

if __name__ == "__main__":
    ori_path = 'data/synthetic/point_distribution/synthetic_duplicated'
    dst_path = 'data/synthetic/ablation1'

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    else:
        all_files = glob.glob(os.path.join(dst_path, '*.[jJpP]*[gG]'))
        for file in all_files:
            os.remove(file)
    
    imgs = glob.glob(os.path.join(ori_path, 'img_*.jpg'))
    for img_path in imgs:
        img = cv.imread(img_path)
        cv.imwrite(dst_path + '/' + os.path.basename(img_path), img)    

    imgs_blur = random.sample(imgs, 15)
    blur_type = ['horizontal', 'vertical']

    for img_path in imgs_blur:
        img = cv.imread(img_path)
        blur = motion_blur(img, 5, random.choice(blur_type))
        
        # Apply random transformations
        angle = random.uniform(-30, 30)
        x_translation = random.uniform(-30, 30)
        y_translation = random.uniform(-15, 15)
        # scale = random.uniform(0.6, 1.4)
        
        # blur = rotate_image(blur, angle)
        blur = translate_image(blur, x_translation, y_translation)
        # blur = scale_image(blur, scale)
        
        # plt.subplot(2, 1, 1)
        # plt.imshow(img)
        # plt.subplot(2, 1, 2)
        # plt.imshow(blur)
        # plt.show()
        # cv.waitKey(0)
        # cv.destroyAllWindows
        
        cv.imwrite(dst_path + '/' + os.path.basename(img_path).replace('.jpg', '_blur.jpg'), blur)