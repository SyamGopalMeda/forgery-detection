import torch
from utils import *
from PIL import Image
import numpy as np
import sys
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detect_forgery(image_path, output_path):
    # Load models
    trans_sim = TransmissionSimulator().to(DEVICE).eval()
    detector = SocialMediaForgeryDetector().to(DEVICE).eval()
    
    trans_sim.load_state_dict(torch.load("models/transmission_simulator.pth", map_location=DEVICE))
    detector.load_state_dict(torch.load("models/forgery_detector.pth", map_location=DEVICE))
    
    # Process image
    x = preprocess_image(image_path).to(DEVICE)
    
    with torch.no_grad():
        # Simulate transmission
        x_noisy = x + trans_sim(x)
        # Detect forgery
        mask = (detector(x_noisy) > 0.5).float()
    
    # Create visualization
    original = Image.open(image_path)
    mask_np = mask.squeeze().cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype('uint8')).resize(original.size)
    
    # Overlay result
    result = original.copy()
    result.putalpha(255)
    overlay = Image.new('RGBA', original.size, (255, 0, 0, 128))
    result.paste(overlay, (0, 0), mask_pil)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to RGB if saving as JPEG, or keep as PNG
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        result = result.convert('RGB')
    
    result.save(output_path)
    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect.py <input_image_path> <output_image_path>")
        sys.exit(1)
    
    result_path = detect_forgery(sys.argv[1], sys.argv[2])
    print(f"Result saved to {result_path}")