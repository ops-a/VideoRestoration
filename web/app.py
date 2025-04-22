import os
import sys
import yaml

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import numpy as np
from src.models.restoration_model import VideoRestorationModel, RainRemovalModel, EncoderDecoderModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Create Flask app with correct static folder configuration
app = Flask(__name__, 
            static_url_path='/static',
            static_folder='static')

# Use absolute paths for upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint, handling different architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model architecture based on checkpoint keys
    if 'bottleneck.2.fc1.weight' in checkpoint:  # SEBlock layer
        model = RainRemovalModel(in_channels=3).to(device)
    elif 'bottleneck.4.weight' in checkpoint:  # EncoderDecoderModel with extra layers
        model = EncoderDecoderModel(in_channels=3).to(device)
    else:
        # Residual Block architecture
        model = VideoRestorationModel(in_channels=3).to(device)
    
    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict format
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# Initialize model with absolute path
checkpoint_path = os.path.join(project_root, 'checkpoints', 'advanced_rain_removal_net.pth')
model = load_model(checkpoint_path)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_metrics(input_path: str, output_path: str) -> dict:
    """Calculate PSNR and SSIM between input and output images"""
    # Load images
    input_img = np.array(Image.open(input_path).convert('RGB'))
    output_img = np.array(Image.open(output_path).convert('RGB'))
    
    # Calculate metrics
    psnr_value = psnr(input_img, output_img, data_range=255)
    ssim_value = ssim(input_img, output_img, data_range=255, channel_axis=2)
    
    return {
        'psnr': round(psnr_value, 2),
        'ssim': round(ssim_value, 4)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f"restored_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        try:
            # Process image
            model.process_image(input_path, output_path)
            
            # Calculate metrics
            metrics = calculate_metrics(input_path, output_path)
            
            return jsonify({
                'success': True,
                'input_image': f'/static/uploads/{filename}',
                'output_image': f'/static/uploads/{output_filename}',
                'metrics': metrics
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 