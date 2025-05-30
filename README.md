# 🌊 Flood_Segmentation and Disaster Analysis

This project is designed to perform flood segmentation using deep learning techniques. It includes a UNet-based segmentation model trained to identify flooded areas from satellite imagery. The system also supports instance segmentation, change detection, and visualization of results through a Flask web application.


## 📁 Project Structure

project/ <br>
├── app.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Flask app for the web interface <br>
├── detect_chages.py &nbsp;&nbsp;&nbsp;# Script for change detection <br>
├── floods_project.ipynb&nbsp;&nbsp;   # Jupyter Notebook for experimentation <br>
├── instace_segmentation.py&nbsp;&nbsp;# Instance segmentation implementation <br>
├── segmentation_model.pth&nbsp;&nbsp; # Trained UNet model <br>
├── unet.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# UNet model architecture <br>
├── utils.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Utility functions <br>
├── dataset/ <br>
│ ├── Image/ &nbsp;&nbsp;&nbsp;&nbsp;# Satellite images <br>
│ ├── Mask/  &nbsp;&nbsp;&nbsp;&nbsp;# Corresponding segmentation masks <br>
│ └── metadata.csv &nbsp;&nbsp;&nbsp;# Image metadata <br>
├── static/ &nbsp;&nbsp;&nbsp;&nbsp; # Static files for Flask <br>
└── templates/&nbsp;&nbsp;&nbsp;&nbsp;# HTML templates for Flask <be>


## 🚀 Features

- Flood segmentation using UNet.
- Instance segmentation support.
- Change detection over time.
- Visualization of predictions via a web interface.
- Organized dataset of images and masks for training/testing.

## 🧠 Model

The project uses a UNet-based model (`unet.py`) trained to segment flooded areas. The trained model is saved in `segmentation_model.pth`.

## 🖥️ Getting Started

### Prerequisites

- Python 3.8+
- Install Flask
    ```bash
       pip install flask
    
- Install Numpy
    ``` bash
     pip install numpy
    
- Install Pandas
    ``` bash
     pip install pandas
    
- Install PIL
     ``` bash
     pip install pillow

- Install CV2
     ``` bash
     pip install cv2

- Install torch and torchvision
     ```bash
     pip install torch
     pip install torchvision

- Install skimage
     ```bash
     pip install scikit-image

## Running the Flask App
-   ```bash
     cd project
     python app.py

# Contributors ✨
  Thanks to these amazing contributors:
   - [swathi-agarwal](https://github.com/swathi-agarwal) - Machine Learning Development to work on the semantic segmentation of the project.
   - [Deepakmity02](https://github.com/Deepakmity02) - Implemented Machine Learning models on jupyter notebook.
   - [HaniCode254](https://github.com/HaniCode254) - Frontend Developement to work on the frontend part of the project.


## Dataset

The dataset is located in (`dataset/`):
- (`Image/`): Raw satellite images.
- (`Mask/`): Binary or multiclass masks indicating flooded regions.
- (`metadata.csv`): Supplementary metadata for analysis.



## Jupyter Notebook
The notebook (`floods_project.ipynb`) provides a step-by-step demonstration of:
- Data preprocessing
- Model training
- Segmentation results
- Visualization

  

## Utilities
(`utils.py`): Includes helper functions for preprocessing, visualization, and mask/image handling.

## Change Detection
Implemented in (`detect_chages.py`), this script compares pre- and post-disaster images to assess flood impact.


## Instance Segmentation
Handled via (`instace_segmentation.py`), this script segments multiple flooded objects using instance-level techniques.


## To-Do
- Improve UI of the web dashboard.
- Add performance metrics (IoU, Dice Score).
- Enhance real-time data handling and upload support.
- Integrate with a larger disaster analysis system.

## License
 This project is for educational and research purposes only.

