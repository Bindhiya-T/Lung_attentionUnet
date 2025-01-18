from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load your trained model
model = load_model('updated_model.h5')

def preprocess_image(img):
    img = img.resize((128, 128)).convert('L')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resized-image' not in request.form:
        return redirect(request.url)

    # Decode the base64 image data
    img_data = request.form['resized-image']
    img_data = img_data.split(',')[1]  # Remove the data URL prefix
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))

    # Preprocess the image
    img_array = preprocess_image(img)

    # Predict the mask
    pred_mask = model.predict(img_array)[0]
    pred_mask = (pred_mask.squeeze() > 0.5).astype(np.float32)

    # Save the original image
    original_filename = 'original.png'
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    img.save(original_path)

    # Save the predicted mask as an image
    mask_filename = 'mask.png'
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
    plt.imsave(mask_path, pred_mask, cmap='gray')

    # Combine the original and predicted mask images side by side
    combined_img = Image.new('RGB', (img.width * 2, img.height))
    combined_img.paste(img, (0, 0))
    combined_mask = Image.open(mask_path).convert('RGB')
    combined_img.paste(combined_mask, (img.width, 0))
    combined_filename = 'combined.png'
    combined_path = os.path.join(app.config['UPLOAD_FOLDER'], combined_filename)
    combined_img.save(combined_path)

    # Determine if the tumor is present
    tumor_present = np.max(pred_mask) > 0.5
    result = "Cancerous" if tumor_present else "Non-Cancerous"

    return render_template('index.html', original_image=combined_path, result=result)

if __name__ == '__main__':
    app.run(debug=True)
