from flask import Flask, render_template, request
import os
from utils import load_model, preprocess_image, predict_mask, overlay_mask, calculate_flood_area, calculate_change_severity, estimate_structures_affected
from instace_segmentation import load_instance_model, predict_instance_segmentation
from detect_chages import detect_changes

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model from .pth
model = load_model('segmentation_model.pth')
instance_model = load_instance_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']
    filename = file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Semantic segmentation

    original, mask = predict_mask(model, image_path)
    result_img = overlay_mask(original, mask)
    result_path = os.path.join(UPLOAD_FOLDER, 'semantic_result.png')
    result_img.save(result_path)


    # Instance segmentation

    inst_result_img = predict_instance_segmentation(instance_model, image_path)
    inst_result_path = os.path.join(UPLOAD_FOLDER, 'instance_result.png')
    inst_result_img.save(inst_result_path)


    return render_template('index.html', 
                           uploaded=True, 
                           original=filename, 
                           semantic_result='semantic_result.png',
                           instance_result='instance_result.png')


@app.route('/change_detection', methods=["POST"])
def change_detection():
    if 'before_image' not in request.files or 'after_image' not in request.files:
        return 'Both images are required for change detection', 400
    
    before_image = request.files['before_image']
    after_image = request.files['after_image']

    before_filename = 'before' + before_image.filename
    after_filename = 'after' + after_image.filename

    before_path = os.path.join(UPLOAD_FOLDER, before_filename)
    after_path = os.path.join(UPLOAD_FOLDER, after_filename)

    before_image.save(before_path)
    after_image.save(after_path)

    # Detect changes

    change_img = detect_changes(before_path, after_path)
    change_result_path = os.path.join(UPLOAD_FOLDER, 'change_result.png')
    change_img.save(change_result_path)

    return render_template('index.html',
                           change_uploaded = True,
                           before_img = before_filename,
                           after_img = after_filename,
                           change_result = 'change_result.png')
if __name__ == '__main__':
    app.run(debug=True)