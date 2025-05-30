# 🌊 Flood_Segmentation and Disaster Analysis

This project is designed to perform flood segmentation using deep learning techniques. It includes a UNet-based segmentation model trained to identify flooded areas from satellite imagery. The system also supports instance segmentation, change detection, and visualization of results through a Flask web application.


## 📁 Project Structure

project/
├── app.py # Flask app for the web interface
├── detect_chages.py # Script for change detection
├── floods_project.ipynb # Jupyter Notebook for experimentation
├── instace_segmentation.py # Instance segmentation implementation
├── segmentation_model.pth # Trained UNet model
├── unet.py # UNet model architecture
├── utils.py # Utility functions
├── dataset/
│ ├── Image/ # Satellite images
│ ├── Mask/ # Corresponding segmentation masks
│ └── metadata.csv # Image metadata
├── static/ # Static files for Flask
└── templates/ # HTML templates for Flask
