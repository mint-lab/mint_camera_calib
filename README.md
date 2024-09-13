## Generalized Camera Calibration: Camera Model Selection and Calibration with Effective Image Sampling
_Generalized Camera Calibration_ is a camera calibration and selection tool that effectively addresses the following 
questions during calibration: 

* What is the best model for camera calibration?
* How many images are needed to calibrate a camera? 
* What are the best viewpoints for taking images for dataset construction?
* Are there any criteria for choosing a camera model among many available camera models?
  
Our framework focuses on three core components: image sampling, camera parameter estimation, and model selection, 
as shown in the figure below:

<p align='center'>
  <img width="800px" src="https://github.com/mint-lab/mint_camera_calib/blob/main/docs/pipeline.png" />
  <br/>
  <i> Generalized Camera Calibration Framework.</i>
</p>

* **_Image Sampling:_**

  Build a robust dataset for calibration by employing techniques to filter out low-quality images
and strategically selecting the most optimal viewpoints for capturing the next images in the dataset.

* **_Camera Calibration:_**

  Propose 22 different camera models by combining various projection and distortion models, and perform comprehensive
  calibration for each to ensure optimal accuracy across a range of configurations.
  
* **_Model Selection:_**
  
  Evaluate each model based on AIC and BIC, two fundamental and widely employed criteria in model selection, 
  aiming to identify the most suitable camera model.
  
## Usage
* __Prerequisite:__
  * If you don't install OpenCV, please install OpenCV: `pip install opencv-python`.

* __Image Sampling:__
  *  `python image_samling.py video_path out_dir [-c config_file.json]`.
        * `-c` (or `--config_file`): Specify a configuration file that can change the _chessboard_pattern_
      (default: `cfgs/config.json`).
        * The results of image sampling will be saved in the `out_dir` folder, making them ready for calibration.
   
* __Camera Calibration and Selection:__
  *  `python cam_cali_select.py img_dir save_dir [-c config_file.json]`.
        * `-c` (or `--config_file`): Specify a configuration file that can change the _chessboard_pattern_, __criteria__
          (AIC or BIC).
      (default: `cfgs/config.json`).
        * Perform calibration on the images in the `img_dir` folder using various camera models. Evaluate the performance
          of each model, select the best one, and save the results in the `save_dir` folder.

* __Visualization:__
  
  In addition to presenting the calibration results, our framework provides three key types of visualizations: model-wise heatmap,
  point-wise heatmap, and camera position visualization.

  *  `python visualize.py result_dir [-t visualization_type] [-c config_file.json]`.
        * `-c` (or `--config_file`): Specify a configuration file that can change visualization configuration.
      (default: `cfgs/config.json`).
        *  Visualize the calibration and selection results in the `results_dir` directory based on the chosen `visualization_type`.
  *  Three types of visualization `visualization_type = `:
      * `model_wise_score` or `model_wise_rms`:
        Enhance model selection by simplifying the comparison process. Choose to visualize either `model_wise_score`, which represents
        scores in heatmaps after applying selection criteria, or `model_wise_rms`, which displays RMSE values for each model post-calibration.
  
        <p align="center">
          <img width="50%" src="https://github.com/mint-lab/mint_camera_calib/blob/main/docs/model_wise_score.png" alt="Model Wise Score" />
          <br />
          <i>An example of model-wise visualization using model_wise_score.</i>
        </p>      
  
      * `point_wise`:
         Identify and address images with large reprojection errors, providing insights for enhancing overall calibration accuracy.
 
        <p align='center'>
          <img width="45%" src="https://github.com/mint-lab/mint_camera_calib/blob/main/docs/point_wise.png" />
          <br />
          <i>An example of point-wise visualization.</i>
        </p>
  
      * `cam_pose`:
        Camera Position Visualization reveals the location of each camera used in image capture within a dataset. It offers valuable insights for optimizing 
        camera placement, ensuring a more diverse and well-distributed array of shots. This enhances the dataset by providing a wider variety of perspectives
        and improving overall coverage.
 
        An example of `cam_pose` accompanied by a visual comparison of dataset construction using random sampling versus our proposed image sampling method:
 
        <p align="center">
            <img src="https://github.com/mint-lab/mint_camera_calib/blob/main/docs/random_dataset_cam_pose.png" width="35%" alt="Random dataset" />
            <img src="https://github.com/mint-lab/mint_camera_calib/blob/main/docs/our_dataset_cam_pose.png" width="35%" alt="Our method dataset" />
            <br>
            <i>(a): Random dataset (left). (b): Our method dataset (right).</i>
        </p>




### Authors
* [Nguyen Cong Quy](https://github.com/ncquy)
* [Sunglok Choi](https://mint-lab.github.io/sunglok/)
