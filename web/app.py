import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import numpy as np
from src.models.restoration_model import VideoRestorationModel # Adjust import based on your actual model location

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model
model = VideoRestorationModel()  # Adjust based on your model initialization
model.eval()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        try:
            # Load and preprocess image
            image = Image.open(filepath).convert('RGB')
            # Add your preprocessing steps here
            
            # Run model inference
            with torch.no_grad():
                restored_image, metrics = model.process_image(image)
            
            # Save restored image
            restored_filename = f'restored_{filename}'
            restored_path = os.path.join(app.config['UPLOAD_FOLDER'], restored_filename)
            restored_image.save(restored_path)
            
            return jsonify({
                'original': filename,
                'restored': restored_filename,
                'metrics': metrics
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 