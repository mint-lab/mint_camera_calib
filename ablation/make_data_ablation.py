import cv2 as cv
import numpy as np
import pandas as pd
import random
import os
from scipy.spatial.transform import Rotation


def gen_chessboard_points(chessboard_pattern, cellsize=0.025, dtype=np.float32):
    '''Generate 3D points with regular interval on a plane (a.k.a. chessboard).'''
    obj_pts = cellsize * np.array([(c, r, 0) for r in range(chessboard_pattern[1]) for c in range(chessboard_pattern[0])], dtype=dtype)
    row, col = obj_pts.shape
    obj_pts = obj_pts.reshape(row, 1, col)
    return obj_pts


def generate_intrinsic_maxtrix(intrinsic_type, img_size):
    if intrinsic_type == 'P0':
        fx = fy = random.randint(800, 1000)
        cx = (img_size[1] - 1) / 2
        cy = (img_size[0] - 1) / 2
    elif intrinsic_type == 'P1':
        fx, fy = random.randint(800, 1000), random.randint(800, 1000)
        cx = (img_size[1] - 1) / 2
        cy = (img_size[0] - 1) / 2
    elif intrinsic_type == 'P2':
        fx = fy = random.randint(800, 1000)
        cx = (img_size[1]) / 2 - 10
        cy = (img_size[0]) / 2 - 10
    elif intrinsic_type == 'P3':
        fx, fy = random.randint(800, 1000), random.randint(800, 1000)
        cx = (img_size[1]) / 2 - 10
        cy = (img_size[0]) / 2 - 10
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


def generate_dist_coef(dist_type):
    if dist_type.startswith('BC'):
        tmp = np.array([-0.2, 0.1, -0.01, 0.005, 0.])
        dist = np.zeros(5)
        if dist_type == 'BC1':
            dist[0] = tmp[0]
        elif dist_type == 'BC2':
            dist[0] = tmp[0]
            dist[1] = tmp[1]
        elif dist_type == 'BC3':
            dist = tmp
    else:
        tmp = np.array([-0.2, 0.1, 0., 0.])
        dist = np.zeros(4)
        if dist_type == 'KB1':
            dist[0] = tmp[0]
        elif dist_type == 'KB2':
            dist = tmp
    return dist


def conv_pose2Rt(ori_xyz, pos_xyz, degrees=True):
    '''Convert the given camera pose to the rotation vector (rvec) and translation vector (tvec).'''
    rvec = -Rotation.from_euler('zyx', ori_xyz[::-1], degrees=degrees).as_rotvec()  # '-' is transpose for 'rvec'.
    tvec = -Rotation.from_rotvec(rvec).apply(pos_xyz)
    return rvec.ravel(), tvec.ravel()


def draw_chessboard(img_pts, img_size, chessboard_pattern, save_path, ind_data):
    """
    Draw a chessboard pattern and save it as an image.
    """
    # Tạo hình ảnh trống với kích thước xác định
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    img.fill(255)  # Fill with white background

    # Vẽ các điểm trên bàn cờ
    for pt in img_pts.squeeze():
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
            cv.circle(img, (x, y), 5, (0, 0, 0), -1)  # Vẽ điểm đen

    # Tạo và vẽ bàn cờ
    for r in range(chessboard_pattern[1]):
        for c in range(chessboard_pattern[0]):
            top_left = (int(c * (img_size[1] / chessboard_pattern[0])),
                        int(r * (img_size[0] / chessboard_pattern[1])))
            bottom_right = (int((c + 1) * (img_size[1] / chessboard_pattern[0])),
                            int((r + 1) * (img_size[0] / chessboard_pattern[1])))
            color = (0, 0, 0) if (r + c) % 2 == 0 else (255, 255, 255)
            cv.rectangle(img, top_left, bottom_right, color, -1)

    # Lưu ảnh
    file_name = os.path.join(save_path, f'chessboard_{ind_data}.png')
    cv.imwrite(file_name, img)


def generate_img(obj_pts, K, dist_coef, dist_type, save_path, img_size, chessboard_pattern):
    ind = 1

    cam_ori = [[-8., -5, 0.], [-10., -5, 0.],
               [-10., -5., 5], [-8., -10., 5.],
               [-8., -5, 15.], [-10., -5, 20.],
               [-8., -8., 10], [-8., -10., 15.]]
    cam_pos = [[0.2, 0.05, -0.35], [0.2, 0.05, -0.4],
               [0.25, 0.1, -0.45], [0.3, 0., -0.5],
               [0.15, 0.05, -0.5]]

    for i in cam_ori:
        for j in cam_pos:
            rvec, tvec = conv_pose2Rt(i, j)
            if dist_type.startswith('BC'):
                img_pts, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist_coef)
            else:
                img_pts, _ = cv.fisheye.projectPoints(obj_pts, rvec, tvec, K, dist_coef)
            img_pts = add_noise(img_pts)

            # Thay thế lưu ảnh bàn cờ
            draw_chessboard(img_pts, img_size, chessboard_pattern, save_path, ind)
            ind += 1


def add_noise(x, mean=0, standard_deviation=1):
    noise = np.random.normal(mean, standard_deviation, x.shape)
    x = x + noise
    x = np.float32(np.round(x, decimals=4))
    return x


def make_data(proj_model, dist_model, data_path, ind):
    model_ind = []
    synthetic_path = []
    cam_model = []
    K_original = []
    dist_original = []
    img_size = (960, 1280)

    for dist_type in dist_model:
        for intrinsic_type in proj_model:
            for _ in range(1):
                K = generate_intrinsic_maxtrix(intrinsic_type, img_size)
                dist_coef = generate_dist_coef(dist_type)

                save_path = os.path.join(data_path, 'model_' + str(ind))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                generate_img(obj_pts, K, dist_coef, dist_type, save_path, img_size, chessboard_pattern)

                model_ind.append(ind)
                synthetic_path.append(save_path)
                cam_model.append({'intrinsic': intrinsic_type, 'dist': dist_type})
                K_original.append(K)
                dist_original.append(dist_coef)
                ind += 1
    return model_ind, synthetic_path, cam_model, K_original, dist_original


if __name__ == '__main__':
    proj_model_KB = ['P3']
    dist_model_KB = ['KB2']
    img_size = (960, 1280)
    chessboard_pattern = (10, 7)
    data_path = 'data/ablation'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    obj_pts = gen_chessboard_points(chessboard_pattern, cellsize=0.035)

    # Generate KB model
    ind = 1
    model_ind_KB, path_KB, cam_model_KB, K_original_KB, dist_original_KB = make_data(
        proj_model_KB, dist_model_KB, data_path, ind)

    data = {'index': model_ind_KB,
            'path': path_KB,
            'K_original': K_original_KB,
            'dist_original': dist_original_KB,
            'ori_model': cam_model_KB}

    df = pd.DataFrame(data)
    df.to_excel(os.path.join(data_path, 'synthetic_data.xlsx'), index=False)
