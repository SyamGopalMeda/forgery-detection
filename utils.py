import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = "datasets"
IMG_SIZE = 512

# Binary mask conversion
class ToBinaryMask:
    def __call__(self, tensor):
        return (tensor > 0.5).float()

# Transmission artifact simulator
class TransmissionSimulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return x + self.process(x)

# Forgery detection model
class SocialMediaForgeryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction with ResNet34
        self.backbone = resnet34(weights='DEFAULT')
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Decoder with proper upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        # Feature extraction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)
        
        # Decode to original size
        output = self.decoder(features)
        return torch.sigmoid(output)

# Dataset loader
class SocialMediaDataset(Dataset):
    def __init__(self, platform_dirs, gt_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            ToBinaryMask()
        ])
        
        for platform in platform_dirs:
            if platform == "Original":
                platform_path = os.path.join(BASE_PATH, "Original")
            else:
                platform_path = os.path.join(BASE_PATH, f"NIST16_{platform}")
                
            gt_path = os.path.join(BASE_PATH, gt_dir)
            
            if not os.path.exists(platform_path):
                print(f"Warning: Platform directory not found - {platform_path}")
                continue
                
            for img_name in os.listdir(platform_path):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    gt_name = img_name.split('.')[0] + '_gt.png'
                    gt_full_path = os.path.join(gt_path, gt_name)
                    if os.path.exists(gt_full_path):
                        self.samples.append((
                            os.path.join(platform_path, img_name),
                            gt_full_path
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        return self.transform(img), self.gt_transform(gt)

# Image preprocessing
def preprocess_image(image_path, size=512):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)