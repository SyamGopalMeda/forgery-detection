import torch
from utils import *
from PIL import Image
import numpy as np
import io

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detect_forgery(img):
    # Load models
    trans_sim = TransmissionSimulator().to(DEVICE).eval()
    detector = SocialMediaForgeryDetector().to(DEVICE).eval()
    
    trans_sim.load_state_dict(torch.load("models/transmission_simulator.pth", map_location=DEVICE))
    detector.load_state_dict(torch.load("models/forgery_detector.pth", map_location=DEVICE))
    
    # Process image
    x = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Simulate transmission
        x_noisy = x + trans_sim(x)
        # Detect forgery
        mask = (detector(x_noisy) > 0.5).float()
    
    # Create visualization
    mask_np = mask.squeeze().cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype('uint8')).resize(img.size)
    
    # Overlay result
    result = img.copy()
    result.putalpha(255)
    overlay = Image.new('RGBA', img.size, (255, 0, 0, 128))
    result.paste(overlay, (0, 0), mask_pil)
    
    return result
