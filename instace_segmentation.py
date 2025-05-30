import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import os


def load_instance_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def predict_instance_segmentation(model, image_path, threshold=0.5, resize=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(resize)

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)


    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']
    masks = predictions['masks'] > 0.5

    image_np = np.array(image)

    for i in range(len(scores)):
        if scores[i] > threshold:
            mask = masks[i, 0].cpu().numpy().astype(np.uint8)
            color = np.random.randint(0, 255, (1, 3), dtype=int).tolist()[0]
            image_np[mask == 1] = image_np[mask == 1] * 0.5 + np.array(color) * 0.5
            x1, y1, x2, y2 = boxes[i].int().cpu().numpy()
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

    result = Image.fromarray(image_np.astype(np.uint8))
    return result
