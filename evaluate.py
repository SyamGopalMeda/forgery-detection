import torch
from torch.utils.data import DataLoader
from utils import (
    TransmissionSimulator, 
    SocialMediaForgeryDetector,
    SocialMediaDataset
)
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = "datasets"
BATCH_SIZE = 4

def evaluate_model():
    # Load trained models
    trans_sim = TransmissionSimulator().to(DEVICE).eval()
    detector = SocialMediaForgeryDetector().to(DEVICE).eval()
    
    try:
        trans_sim.load_state_dict(torch.load("models/transmission_simulator.pth", map_location=DEVICE))
        detector.load_state_dict(torch.load("models/forgery_detector.pth", map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"Error loading models: {str(e)}")
        return

    # Prepare test dataset
    try:
        test_dataset = SocialMediaDataset(
            ["Facebook", "Wechat", "Weibo", "Whatsapp"],
            "NIST16_GT"
        )
        print(f"Found {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluation metrics
    metrics = {
        'iou': 0,
        'precision': 0, 
        'recall': 0,
        'f1': 0,
        'samples_processed': 0
    }

    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, (x, y_true) in enumerate(tqdm(test_loader, desc="Processing batches")):
            x, y_true = x.to(DEVICE), y_true.to(DEVICE)
            
            # Add simulated noise
            x_noisy = x + trans_sim(x)
            
            # Get predictions
            y_pred = detector(x_noisy)
            y_pred = (y_pred > 0.5).float()
            
            # Resize prediction to match ground truth
            y_pred = F.interpolate(y_pred, size=y_true.shape[2:], 
                                 mode='bilinear', align_corners=False)
            
            # Calculate metrics
            intersection = (y_pred * y_true).sum()
            union = (y_pred + y_true).sum() - intersection
            
            metrics['iou'] += intersection.item()
            metrics['precision'] += (intersection / (y_pred.sum() + 1e-6)).item()
            metrics['recall'] += (intersection / (y_true.sum() + 1e-6)).item()
            metrics['samples_processed'] += x.shape[0]

    # Average metrics
    num_batches = len(test_loader)
    metrics['iou'] /= num_batches
    metrics['precision'] /= num_batches
    metrics['recall'] /= num_batches
    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                   (metrics['precision'] + metrics['recall'] + 1e-6)

    print("\n=== Evaluation Results ===")
    print(f"Samples Processed: {metrics['samples_processed']}")
    print(f"IoU (Jaccard Index): {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    evaluate_model()