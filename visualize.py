import os
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
import matplotlib.patches as patches
from cam_cali_select import CameraCalibration

class Visualization():
    def __init__(self, result_path, config_file, visual_type) -> None:
        self.result_path = result_path
        self.config_file = config_file
        self.visual_type = visual_type
        self.config = self.get_default_config()

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.config.update(config)
        except:
            print('Visualization will use the default configuration')

    def get_default_config(self):
        config = {}

        config['proj_model_BC'] = ['P1', 'P2', 'P3', 'P4']
        config['dist_model_BC'] = ['BC0', 'BC1', 'BC2', 'BC4']
        config['proj_model_KB'] = ['P2', 'P4']
        config['dist_model_KB'] = ['KB0', 'KB1', 'KB2']
        return config

    def normalize(self, array):
        min_val = np.min(array)
        max_val = np.max(array)
        array = (array - min_val) / (max_val - min_val)

        return np.round(array, 4)
    
    def normalize_df(self, df):
        return (df - df.stack().min()) / (df.stack().max() - df.stack().min())

    def calculate_luminance(self, color):
        """Calculate the relative luminance of a color."""
        r, g, b, _ = color
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def get_contrasting_color(self, color):
        """Return black or white depending on the luminance of the input color."""
        luminance = self.calculate_luminance(color)
        return 'black' if luminance > 0.5 else 'white'

    def visualize_model_wise(self, df, decimal_place):
        sns.set(font_scale=0.85)  # for label size
        
        cmap = sns.color_palette(self.config['model_wise_cmap'], as_cmap=True)
        # Create a heatmap with values capped at 1 for the colormap
        ax = sns.heatmap(df, annot=True, cmap=cmap, square=True, cbar=True, 
                        vmin=0, vmax=1, cbar_kws={"fraction": 0.027}, fmt=decimal_place)
        
        # Update text colors 
        for text in ax.texts:
            # Find the position of the text
            row, col = int(text.get_position()[1]), int(text.get_position()[0])
            # Get the face color of the cell
            facecolor = ax.collections[0].get_facecolor()[row * df.shape[1] + col]
            # Set text color to contrast with the cell color
            text.set_color(self.get_contrasting_color(facecolor))
        
        # Set the face color for NaN cells to match the background
        for i in range(len(df)):
            for j in range(len(df.columns)):
                if pd.isna(df.iloc[i, j]):
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=True, color='white', linewidth=0))
        
        plt.show()

    def visualize_point_wise(self, error, img_names, square_cell=True):
        # Extract the image index from image names by splitting the names by '_'
        # and taking the last part (assumed to be a numeric identifier).
        img_ind = [img_name.split('_')[-1] for img_name in img_names]

        # Create a dictionary mapping the image indices to the corresponding error values.
        ind2error_dict = dict(zip(img_ind, error))
        
        # Sort the image indices and use them to reorder the error values.
        sorted_img_ind = sorted(ind2error_dict.keys(), key=int)
        sorted_error = [ind2error_dict[key] for key in sorted_img_ind]

        # If square_cell is False, plot without ensuring that each cell is square.
        if not square_cell:
            # Plot the error matrix as a heatmap with automatic aspect ratio adjustment.
            plt.imshow(error, self.config['point_wise_cmap'], interpolation='nearest', aspect='auto')
            
            # Add a colorbar to visualize the error value scale.
            plt.colorbar()  
            
            # Add a title and labels for the x and y axes.
            plt.title('Point-wise RMSE')  
            plt.xlabel('Point Index') 
            
            # Set the ticks for the x-axis, spacing every 5 points.
            plt.xticks(ticks=np.arange(0, error.shape[1], 5))
            plt.ylabel('Image Index')
            
            # Set the y-axis ticks and labels to the image indices.
            plt.yticks(ticks=np.arange(0, error.shape[0]), labels=img_ind) 
        
        # If square_cell is True, ensure each cell in the heatmap is a square.
        else:
            # Create a figure and axis, adjusting the figure size to 10x6.
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the sorted error matrix with an 'equal' aspect ratio to ensure square cells.
            cax = ax.imshow(np.array(sorted_error), cmap=self.config['point_wise_cmap'], interpolation='nearest', aspect='equal')

            # Add a colorbar with a slight reduction in size (shrink=0.9) and position adjustments.
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.0135, pad=0.04, shrink=0.9)
            
            # Set a common font size for axis labels and ticks.
            font_size = 14

            # Label the x-axis and y-axis with appropriate font size.
            ax.set_xlabel('Point Index', fontsize=font_size)
            ax.set_ylabel('Image Index', fontsize=font_size)
            
            # Set the y-ticks to be based on the sorted image indices and adjust the font size.
            ax.set_yticks(np.arange(0, len(sorted_img_ind)))
            ax.set_yticklabels(sorted_img_ind, fontsize=font_size)
            
            # Increase the font size for x and y tick labels.
            ax.tick_params(axis='x', labelsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size)
            
            # Adjust the colorbar's tick label font size to match the axis labels.
            cbar.ax.tick_params(labelsize=font_size)

        # Automatically adjust the layout to ensure that everything fits within the figure.
        plt.tight_layout()
        
        # Display the plot.
        plt.show()

    def visualize_cam_pose(self, obj_pt, rvecs, tvecs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw chessboard points
        ax.scatter(obj_pt[:, 0], obj_pt[:, 1], obj_pt[:, 2], s=50, c='r', marker='+')

        # Draw the camera positions
        for rvec, tvec in zip(rvecs, tvecs):
            # Convert rotation vector to rotation matrix
            R, _ = cv.Rodrigues(rvec)
            
            # Camera position in world coordinates
            camera_position = -R.T @ tvec    

            # Draw a line from the camera center pointing along the z-axis of the camera
            # ax.scatter(camera_position[0], camera_position[1], camera_position[2], s=100, c='b', marker='^')
            # z_axis = R.T @ np.array([[0, 0, 1]]).T
            # ax.quiver(camera_position[0], camera_position[1], camera_position[2],
            #         z_axis[0], z_axis[1], z_axis[2], length=1, color='b')

            # Draw the camera's orientation (X, Y, Z axes)
            x_axis = R.T @ np.array([[1, 0, 0]]).T
            y_axis = R.T @ np.array([[0, 1, 0]]).T
            z_axis = R.T @ np.array([[0, 0, 1]]).T   

            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                    x_axis[0], x_axis[1], x_axis[2], color='r', length=1, label='X axis')
            
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                    y_axis[0], y_axis[1], y_axis[2], color='g', length=1, label='Y axis')
            
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                    z_axis[0], z_axis[1], z_axis[2], color='b', length=1, label='Z axis')
        
        # # Add coordinate frame at world origin
        # origin = np.array([0, 0, 0])
        # ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='black', length=1.5)
        # ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='black', length=1.5)
        # ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='black', length=1.5)
        # ax.text(0, 0, 0, 'Origin', color='black')

        # Label the axes
        ax.set_xlabel('X world')
        ax.set_ylabel('Y world')
        ax.set_zlabel('Z world')

        plt.show()

    def run_visualize(self):
        try:
            with open(self.result_path, 'r') as f:
                results = json.load(f)

            if self.visual_type == 'model_wise_score':
                model_wise_df = results['model_wise_score']
                model_wise_df = pd.DataFrame.from_dict(model_wise_df, orient='index')
                self.visualize_model_wise(self.normalize_df(model_wise_df), decimal_place=self.config['decimal_place'])
            elif self.visual_type == 'model_wise_rms':
                model_wise_df = results['model_wise_rms']
                model_wise_df = pd.DataFrame.from_dict(model_wise_df, orient='index')
                self.visualize_model_wise(model_wise_df, decimal_place=self.config['decimal_place'])
            elif self.visual_type == 'point_wise':
                camera_calibration = CameraCalibration(results['data_path'])
                imgs, img_name = camera_calibration.load_img()
                img_pts, img_size = camera_calibration.find_chessboard_corners(imgs, self.config['chessboard_pattern'])
                obj_pts = camera_calibration.generate_obj_pts(self.config['chessboard_pattern'], img_pts)
                K = np.array(results["K"])
                dist_coef = np.array(results["dist_coef"])
                rvecs = tuple(np.array(rv) for rv in results["rvecs"])
                tvecs = tuple(np.array(rv) for rv in results["tvecs"])

                pt_wwise_error = []
                for i in range(len(obj_pts)):
                    reproj_img_pts = camera_calibration.find_reproject_points(obj_pts=obj_pts[i], rvecs=rvecs[i], tvecs=tvecs[i], K=K, dist_coef=dist_coef, dist_type=results['score_best_model']['dist_model'])
                    diff = reproj_img_pts - img_pts[i]
                    pt_wwise_error.append(np.array([cv.norm(d, cv.NORM_L2) for d in diff]))
                
                pt_wise_error = np.array(pt_wwise_error)
                pts = [f'pt{i}' for i in range(1, self.config['chessboard_pattern'][0] *  self.config['chessboard_pattern'][1] + 1)]
                # point_wise_rmse = pd.DataFrame(pt_wise_error, columns=pts, index=img_name)
                file_name = [file.split('.')[0] for file in results['img_name']]
                self.visualize_point_wise(pt_wise_error, file_name)
            elif self.visual_type == 'cam_pose':
                rvecs = tuple(np.array(rv) for rv in results["rvecs"])
                tvecs = tuple(np.array(rv) for rv in results["tvecs"])
                obj_pt = np.zeros((self.config['chessboard_pattern'][0] * self.config['chessboard_pattern'][1], 3), np.float32)
                obj_pt[:, :2] = np.mgrid[0:self.config['chessboard_pattern'][0], 0:self.config['chessboard_pattern'][1]].T.reshape(-1, 2)
                self.visualize_cam_pose(obj_pt, rvecs, tvecs)
        except:
            print('can not find the result file. Check file name or perform calibration and model selection before visualization!!!')


if __name__ == '__main__':
    # Add arguments thi the parser
    parser = argparse.ArgumentParser(prog='visualize', description='Visualization of model wise score or point wise rmse')
    parser.add_argument('result_file', type=str, help='specify the result file path')
    parser.add_argument('-t', '--type_visualization', default='model_wise_score', type=str, help='specify if user want to display model wise score or point wise rmse')
    parser.add_argument('-c', '--config_file', default='cfgs/config.json', type=str, help='specify a configuration file')

    # Parse the command-line arguments
    args = parser.parse_args()
    visual = Visualization(args.result_file, args.config_file, args.type_visualization)
    visual.run_visualize()