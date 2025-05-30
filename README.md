# 🌊 Flood_Segmentation and Disaster Analysis

This project is designed to perform flood segmentation using deep learning techniques. It includes a UNet-based segmentation model trained to identify flooded areas from satellite imagery. The system also supports instance segmentation, change detection, and visualization of results through a Flask web application.


## 📁 Project Structure

project/ <br>
├── app.py                  &nbsp   # Flask app for the web interface <br>
├── detect_chages.py             # Script for change detection <br>
├── floods_project.ipynb         # Jupyter Notebook for experimentation <br>
├── instace_segmentation.py      # Instance segmentation implementation <br>
├── segmentation_model.pth       # Trained UNet model <br>
├── unet.py                      # UNet model architecture <br>
├── utils.py                     # Utility functions <br>
├── dataset/ <br>
│ ├── Image/                     # Satellite images <br>
│ ├── Mask/                      # Corresponding segmentation masks <br>
│ └── metadata.csv               # Image metadata <br>
├── static/                      # Static files for Flask <br>
└── templates/                   # HTML templates for Flask <br>
