"""
Flask ML Server for Van Gogh Art Restoration
Handles image restoration inference requests
"""

import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from model import BrushstrokeAwareGenerator

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'baseline_best.pth')
# Force CPU to avoid MPS 'Invalid buffer size' error for high-res attention
DEVICE = torch.device('cpu')
TARGET_SIZE = (512, 512)

# Global model
model = None


def load_model():
    """Load the trained model"""
    global model
    print(f"Loading model on {DEVICE}...")
    
    model = BrushstrokeAwareGenerator(in_channels=4, out_channels=3)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model


def preprocess_image(image_bytes, mask_bytes=None):
    """Preprocess image for model input"""
    # Load and resize image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    original_size = image.size
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(image) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
    
    # Handle mask
    if mask_bytes:
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        mask = mask.resize(TARGET_SIZE, Image.NEAREST)
        mask_array = np.array(mask) / 255.0
    else:
        # Auto-detect damaged regions (black pixels)
        gray = np.mean(img_array, axis=2)
        mask_array = (gray < 0.05).astype(np.float32)
    
    mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0)
    
    return img_tensor.to(DEVICE), mask_tensor.to(DEVICE), original_size


def postprocess_image(tensor, original_size):
    """Convert model output to image"""
    output = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(output)
    image = image.resize(original_size, Image.LANCZOS)
    return image


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'device': str(DEVICE)})


@app.route('/predict', methods=['POST'])
def predict():
    """Restore a damaged painting"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image and optional mask
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        mask_bytes = None
        if 'mask' in request.files:
            mask_bytes = request.files['mask'].read()
        
        # Preprocess
        img_tensor, mask_tensor, original_size = preprocess_image(image_bytes, mask_bytes)
        
        # Inference
        with torch.no_grad():
            restored = model(img_tensor, mask_tensor)
        
        # Postprocess
        result_image = postprocess_image(restored, original_size)
        
        # Return as PNG
        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/samples', methods=['GET'])
def get_samples():
    """Get list of sample images"""
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    if not os.path.exists(samples_dir):
        return jsonify({'samples': []})
    
    samples = []
    for f in os.listdir(samples_dir):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            samples.append({
                'name': f,
                'url': f'/sample/{f}'
            })
    
    return jsonify({'samples': samples[:10]})


@app.route('/sample/<filename>', methods=['GET'])
def get_sample(filename):
    """Serve a sample image"""
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    filepath = os.path.join(samples_dir, filename)
    
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    return jsonify({'error': 'Not found'}), 404


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=False)
