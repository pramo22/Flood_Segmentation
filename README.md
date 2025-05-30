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
- Install required libraries:
  ```bash
  pip install cv2
  
```bash
  pip install torch





