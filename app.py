import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import base64
from io import BytesIO

# --------------------------
# Flask App Setup
# --------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------
# Rebuild CNN Model
# --------------------------
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
conv_base.trainable = False

cnn_model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.load_weights('best_model.weights.h5')

# --------------------------
# Activity Map
# --------------------------
activity_map = {
    'c0': 'Safe driving',
    'c1': 'Texting - right',
    'c2': 'Talking on the phone - right',
    'c3': 'Texting - left',
    'c4': 'Talking on the phone - left',
    'c5': 'Operating the radio',
    'c6': 'Drinking',
    'c7': 'Reaching behind',
    'c8': 'Hair and makeup',
    'c9': 'Talking to passenger'
}

# --------------------------
# MediaPipe Pose Setup
# --------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --------------------------
# Image Preprocessing
# --------------------------
def preprocess_image(img, target_size=(224,224)):
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized / 255.0
    return np.expand_dims(img_norm, axis=0)

# --------------------------
# Prediction Function
# --------------------------
def predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # CNN Prediction
    img_input = preprocess_image(img)
    cnn_pred = cnn_model.predict(img_input)
    class_idx = np.argmax(cnn_pred, axis=1)[0]
    predicted_class = activity_map[f'c{class_idx}']

    # Pose detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # Copy image and draw landmarks if detected
    img_vis = img_rgb.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img_vis, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))

    # Save plotted image to static folder as base64
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img_vis)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    buf.close()
    plot_img = f"data:image/png;base64,{encoded}"

    return predicted_class, plot_img

# --------------------------
# Flask Routes
# --------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result, plot_img = predict(file_path)
            if result is None:
                return jsonify({'error': 'Error processing image'}), 500
            return jsonify({'prediction': result, 'plot_img': plot_img})
    return render_template('index.html')

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
