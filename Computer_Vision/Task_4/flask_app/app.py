from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained U-Net model
MODEL_PATH = r'D:\Projects\Software_Engineering\Artificial_Intelligence\Cellula_Internship\Computer_Vision\Task_5\flask_app\Water_Segmentation_v2.0.keras'

if not os.path.isfile(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
else:
    print("Model file found.")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# Function to preprocess input image
def preprocess_image(image: Image.Image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image (0, 1)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to post-process model output
def postprocess_output(output):
    mask = (output > 0.5).astype(np.uint8)  # Convert to binary mask (threshold = 0.5)
    mask = np.squeeze(mask)  # Remove extra dimensions
    mask = Image.fromarray(mask * 255)  # Convert back to image
    return mask

# Define route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Define route for water segmentation
@app.route('/segment', methods=['POST'])
def segment_water():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Read the image from the file
    img = Image.open(file.stream)

    # Preprocess the image for model input
    processed_image = preprocess_image(img)

    # Perform prediction using the U-Net model
    prediction = model.predict(processed_image)

    # Post-process the model output to get the segmentation mask
    mask = postprocess_output(prediction)

    # Save the mask to a buffer to send it as a response
    mask_io = io.BytesIO()
    mask.save(mask_io, 'PNG')
    mask_io.seek(0)

    return send_file(mask_io, mimetype='image/png')

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
