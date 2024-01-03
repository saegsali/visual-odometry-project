# Visual Odometry Project
This is a project for the course [Vision Algorithms for Mobile Robotics](https://rpg.ifi.uzh.ch/teaching.html#VAMR) of the University of Zurich given by Prof. Dr. Davide Scaramuzza in the fall semester 2023.  
The goal of this project is to implement a simple, monocular, visual odometry (VO) pipeline with the most essential features: initialization of 3D landmarks, keypoint tracking between two frames, pose estimation using established 2D ↔ 3D correspondences, and triangulation of new land- marks.

## Installation
1. **Clone the repository**:  
    First, clone the repository either using SSH or HTTPS:
    Using SSH:

    ```
    git clone git@github.com:saegsali/visual-odometry-project.git
    ```
    Using HTTPS:
    ```
    git clone https://github.com/saegsali/visual-odometry-project.git
    ```
2. **Create a conda environment with the necessary dependencies**:  
    ```
    cd visual-odometry-project
    conda env create -f environment.yml
    conda activate visual-odometry
    ```
3. **Install the Visual Odometry Package:**
Install in editable mode using `-e` flag.
    ```
    pip install -e . 
    ```

### Dataset
You can use the setup script to download the dataset:
```
chmod +x setup.sh
./setup.sh
```
This will download all the datasets and extract them to the `data` folder.

## Usage
To run the visual odometry pipeline, run the following commands:
```
conda activate visual-odometry
python src/main.py
```

## Recordings
Recordings of the pipeline for different datasets can be found here:
- Malaga Urban Dataset: https://youtu.be/
- KITTI Dataset: https://youtu.be/
- Parking Dataset: https://youtu.be/

The recordings were made on a MacBook Air (2022) with the following specs:
- Processor: Apple M2 8-Core CPU (3.5 GHz for 4 performance cores)
- Memory: 16 GB (LPDDR5)

Our implementation of the visual odometry pipeline as of now is not particularly optimized for speed and uses just one thread. A significant amount of time is spent on plotting the point cloud and the trajectory.  
Note that performance and accuracy of the pipeline depends on the tracking algorithm used. In the recordings above, the SIFT feature detector and tracker was used.

## File Structure
The project consists of the following files and folders:
```
src
├── main.py                 # Main file to run the visual odometry pipeline
└── vo                      # Visual Odometry Package
    ├── __init__.py
    ├── algorithms          # Contains different algorithms used in the pipeline
    │   ├── __init__.py
    │   └── ransac.py       # RANSAC algorithm. Used for outlier rejection.
    ├── features            # Contains feature detection and tracking algorithms
    │   ├── __init__.py
    │   ├── harris.py       # Harris Corner Detector
    │   ├── klt.py          # KLT Tracker
    │   ├── sift.py         # SIFT Feature Detector
    │   └── tracker.py      # Wrapper for feature tracking. Can be used to switch between different trackers.
    ├── helpers.py          # Helper functions
    ├── landmarks           # Contains landmark triangulation algorithms
    │   ├── __init__.py
    │   └── triangulation.py # Triangulation algorithm
    ├── pose_estimation     # Contains pose estimation algorithms
    │   ├── __init__.py
    │   └── p3p.py          # P3P Pose Estimation algorithm
    ├── primitives          # Contains primitive classes used in the pipeline
    │   ├── __init__.py
    │   ├── features.py     # Feature class. Used to store feature information like keypoints, descriptors, etc.
    │   ├── frame.py        # Frame class. Used to store frame information the image, features, etc.
    │   ├── loader.py       # Dataset loader. See Dataset section on how to get the datasets.
    │   ├── matches.py      # Matches class. Used to store matches between two frames.
    │   └── state.py        # State class. Used to store the current state of the pipeline. E.g. current frame, previous frame, etc.
    ├── sensors             # Contains sensor models
    │   ├── __init__.py
    │   └── camera.py       # Camera class
    └── visualization       # Contains visualization functions
        ├── __init__.py
        ├── overlays.py     # Contains functions to overlay different visualizations on top of the image
        └── point_cloud.py  # Contains functions to visualize the point cloud
tests                       # Contains unit tests (pytest)
├── test_data
│   └── ...
├── test_features.py
├── test_harris.py
├── test_helpers.py
├── test_p3p.py
├── test_ransac.py
└── test_triangulation.py
```
