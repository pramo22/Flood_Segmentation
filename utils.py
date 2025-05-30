import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from unet import UNet
import cv2
from skimage.metrics import structural_similarity as ssim

def load_model(model_path):
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0), image

def predict_mask(model, image_path):
    input_tensor, original = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)[0][0]
        mask = (output > 0.3).numpy().astype(np.uint8) * 255  # Use threshold 0.3 for better detection
    return original, mask

def overlay_mask(image, mask):
    mask_img = Image.fromarray(mask).convert("L").resize(image.size)
    red_mask = Image.new("RGB", image.size, (255, 0, 0))
    result = Image.composite(red_mask, image, mask_img)  # Better visibility than blend
    return result


def calculate_flood_area(mask, pixel_area_m2=1.0):
    FLOODED_CLASS_INDEX = 1
    flooded_pixels = (mask == FLOODED_CLASS_INDEX).sum()
    area_m2 = flooded_pixels * pixel_area_m2
    area_km2 = area_m2 / 1_000_000
    return round(area_km2, 2)



def estimate_structures_affected(mask):
    STRUCTURE_CLASS_INDEX = 2 
    return int((mask == STRUCTURE_CLASS_INDEX).sum() / 1000)


def calculate_change_severity(before_path, after_path):
    before = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)

    if before is None or after is None:
        raise FileNotFoundError("One of the input images could not be read.")
        
    before = cv2.resize(before, (after.shape[1], after.shape[0]))

    score, _ = ssim(before, after, full=True)

    return round((1 - score) * 100, 2)