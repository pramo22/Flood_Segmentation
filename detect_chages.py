from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import cv2 as  cv

def detect_changes(image1_path, image2_path):
    img1 = cv.imread(image1_path)
    img2 = cv.imread(image2_path)

    # Resize after image to match before image
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Optional threshold highlight changes
    _, thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV, cv.THRESH_OTSU)

    # Convert to color for visualization 
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(img2, (x,y), (x + w, y + h), (0, 0, 255), 2)

    result_img = Image.fromarray(cv.cvtColor(img2, cv.COLOR_BGR2RGB))

    return result_img