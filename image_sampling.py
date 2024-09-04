import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import hamming
import glob
import json
import argparse


class ImageSampling:
    def __init__(self, video_path, output_dir, config_file='cfgs/config.json') -> None:
        self.video_path = video_path
        self.output_dir = output_dir
        self.config_file = config_file
        self.config = self.get_default_config()

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.config.update(config)
        except:
            print('Image sampling will use the default configuratioin')

    @staticmethod
    def get_default_config():
        config = {}
        config['cell_size'] = 30
        config['resolution_levels'] = [2, 3, 4]
        config['chessboard_pattern'] = (10, 7)
        config['distance_threshold'] = 0.12
        config['blur_threshold'] = 80
        config['multi_score_threshold'] = 1000

        config['key_pause'] = ord(' ')
        config['key_exit'] = 27

        config['text_color'] = (0, 0, 255)
        config['status_font'] = cv.FONT_HERSHEY_DUPLEX
        config['status_font_scale'] = 0.6
        config['status_offset'] = (10, 25)
                
        return config

    def make_clean_directory(self, directory_path):
        """
        Creates a new directory or cleans an existing one by removing all files 
        with extensions that match the pattern *.jpg, *.jpeg, *.JPG, *.JPEG, etc.

        Args:
            directory_path (str): Path to the directory to be created or cleaned.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        else:
            # Find all files matching the pattern *.[jJpP]*[gG] and remove them
            image_files = glob.glob(os.path.join(directory_path, '*.[jJpP]*[gG]'))
            for image_file in image_files:
                os.remove(image_file)

    def laplacian_blur_metric(self, img):
        """
        Computes a blur score for the given image using the Laplacian operator.
        
        Args:
            img (numpy.ndarray): The input image in BGR format.
            
        Returns:
            float: A blur score indicating the sharpness of the image. 
                Higher values indicate a sharper image, while lower values indicate more blur.
        """
        # Convert image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Compute the Laplacian (second derivative)
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        
        # Calculate blur score (mean of squared Laplacian values)
        blur_score = np.mean(laplacian ** 2)
        
        return blur_score

    def find_chessboard_corners(self, img):
        # Find 2D corner points from given images
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, self.config['chessboard_pattern'])

        corners = None
        if complete:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)

        return corners

    def multi_resolution_score(self, descriptor, resolution_level):
        """
        Calculates a multi-resolution score for an image descriptor.

        Args:
            descriptor (list or array-like): A collection of numeric values representing features of an image.
            resolution_level (int): The resolution level at which the descriptor is calculated. Higher values represent higher resolutions.
            
        Returns:
            float: A score calculated as the sum of the descriptor multiplied by a scaling factor based on the resolution level.
        """
        return sum(descriptor) * (2 ** resolution_level)

    def multi_resolution_descriptor(self, img, resolution_level=None, cell_size=None):
        corners = self.find_chessboard_corners(img) 
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

    def run_img_sampling(self):
        """
        Selects images from a video file based on certain criteria such as blur, hamming distance, and multi-resolution score.

        Args:
            video_file_path (str): Path to the input video file.
            output_directory (str): Directory where selected images will be saved.
            cell_size (tuple): Size of the cells for descriptor computation.
            resolution_levels (list): List of resolution levels to compute descriptors.
            chessboard_pattern (tuple): Chessboard pattern size for corner detection.
            distance_threshold (float): Minimum hamming distance threshold between images.
            blur_threshold (float): Maximum allowed blur threshold for an image.
            multi_score_threshold (float): Minimum multi-resolution score threshold to select an image.

        Returns:
            list: A list of selected images.
        """
        
        # Initialize window for displaying images
        window_name = 'Image Selection'
        video_capture = cv.VideoCapture(self.video_path)
        assert video_capture.isOpened(), "Error: Unable to open video file."

        # Initialize lists to store descriptors and selected images
        all_descriptors = []
        selected_images = []
        selected_image_count = 0
        
        # Create output folder for saving selected images
        self.make_clean_directory(self.output_dir)

        while True:
            # Capture an image from the video
            valid_frame, frame = video_capture.read()
            if not valid_frame:
                break

            # Display the current frame
            cv.putText(frame, f'Number of Selected Images: {selected_image_count}', self.config['status_offset'], self.config['status_font'], self.config['status_font_scale'], self.config['text_color'])
            cv.imshow(window_name, frame)

            # Process key events for image selection
            key = cv.waitKey(10)
            if key == self.config['key_pause']:  # Space: Pause and show corners
                key = cv.waitKey()
                if key == ord('\r'):  # Enter: Select image if it meets criteria
                    if self.laplacian_blur_metric(frame) > self.config['blur_threshold']:  # Check if the image is not blurry
                        if len(selected_images) == 0:
                            selected_image_count += 1
                            image_save_path = os.path.join(self.output_dir, f'img_{selected_image_count}.jpg')
                            cv.imwrite(image_save_path, frame)
                            selected_images.append(frame)
                            all_descriptors.append(self.multi_resolution_descriptor(frame, cell_size=self.config['cell_size']))
                        else:
                            # Calculate hamming distances between current frame descriptor and all previous descriptors
                            new_frame_descriptor = self.multi_resolution_descriptor(frame, cell_size=self.config['cell_size'])
                            hamming_distances = [round(hamming(new_frame_descriptor, descriptor), 2) for descriptor in all_descriptors]
                            # print('==========================All Distances==========================')
                            # print(hamming_distances)

                            # Calculate multi-resolution score for the new frame
                            new_frame_multi_descriptors = [self.multi_resolution_descriptor(frame, resolution_level=resolution_level) for resolution_level in self.config['resolution_levels']]
                            multi_resolution_score_total = sum(self.multi_resolution_score(descriptor, resolution_level) for descriptor, resolution_level in zip(new_frame_multi_descriptors, self.config['resolution_levels']))
                            # print('===========================All Scores============================')
                            # print(multi_resolution_score_total)

                            # Select the image if it meets the distance and score thresholds
                            if all(distance > self.config['distance_threshold'] for distance in hamming_distances) and multi_resolution_score_total > self.config['multi_score_threshold']:
                                selected_image_count += 1
                                image_save_path = os.path.join(self.output_dir, f'img_{selected_image_count}.jpg')
                                cv.imwrite(image_save_path, frame)
                                selected_images.append(frame)
                                all_descriptors.append(new_frame_descriptor)
                                
            if key == self.config['key_exit']:  # ESC: Exit and complete image selection
                break

        cv.destroyAllWindows()
        return selected_images


if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser(prog='image_sampling', description='Image sampling from video')
    parser.add_argument('video_path', type=str, help='specify the video file path')
    parser.add_argument('out_dir', type=str, help='specify the output dir to save images')
    parser.add_argument('-c', '--config_file', default='cfgs/config.json')

    args = parser.parse_args()
    img_selection = ImageSampling(args.video_path, args.out_dir)
    img_selection.run_img_sampling()