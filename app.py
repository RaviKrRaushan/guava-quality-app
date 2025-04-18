from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("best_mobilenetv2.h5")
class_labels = ['rotten', 'mid', 'fresh']

# Folder for uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image prediction
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Video prediction (sample every 10th frame)
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            img = cv2.resize(frame, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img, verbose=0)
            pred_class = class_labels[np.argmax(pred)]
            predictions.append(pred_class)
        frame_count += 1

    cap.release()
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            result = predict_video(file_path)
        elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            result, confidence = predict_image(file_path)
            result = f"{result} (Confidence: {confidence:.2f}%)"
        else:
            return "Unsupported file type", 400

        os.remove(file_path)
        return result
    else:
        return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True)