import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils import *

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = "datasets"
BATCH_SIZE = 4  # Reduced for stability
EPOCHS = 50
IMG_SIZE = 512  # Fixed size for all images

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
            ToBinaryMask()  # Using class instead of lambda
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
                    else:
                        print(f"Warning: Ground truth not found for {img_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        return self.transform(img), self.gt_transform(gt)

def train_transmission_simulator():
    print("Initializing transmission simulator training...")
    try:
        dataset = SocialMediaDataset(["Original", "Facebook"], "NIST16_GT")
        print(f"Found {len(dataset)} training samples")
        
        # Disable multiprocessing for Windows compatibility
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        model = TransmissionSimulator().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            for orig, transmitted in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                orig, transmitted = orig.to(DEVICE), transmitted.to(DEVICE)
                optimizer.zero_grad()
                simulated = model(orig)
                loss = F.l1_loss(simulated, transmitted - orig)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "models/transmission_simulator.pth")
            
            print(f"Transmission Sim Loss: {avg_loss:.6f}")
    except Exception as e:
        print(f"Error in training transmission simulator: {str(e)}")
        raise

def train_forgery_detector():
    print("Initializing forgery detector training...")
    try:
        platforms = ["Facebook", "Wechat", "Weibo", "Whatsapp"]
        dataset = SocialMediaDataset(platforms, "NIST16_GT")
        print(f"Found {len(dataset)} training samples")
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Disable multiprocessing for Windows compatibility
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
        
        trans_sim = TransmissionSimulator().to(DEVICE).eval()
        trans_sim.load_state_dict(torch.load("models/transmission_simulator.pth", map_location=DEVICE))
        
        model = SocialMediaForgeryDetector().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_iou = 0
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                with torch.no_grad():
                    x_noisy = x + trans_sim(x)
                
                optimizer.zero_grad()
                pred = model(x_noisy)
                
                # Ensure same size for loss calculation
                if pred.size() != y.size():
                    pred = F.interpolate(pred, size=y.size()[2:], mode='bilinear', align_corners=False)
                
                loss = F.binary_cross_entropy(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_iou = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                    x_noisy_val = x_val + trans_sim(x_val)
                    pred_val = model(x_noisy_val)
        
                    # Resize prediction to match ground truth
                    pred_val = F.interpolate(pred_val, size=y_val.size()[2:], mode='bilinear', align_corners=False)
        
                    # Convert to binary masks
                    pred_binary = (pred_val > 0.5).float()
                    y_binary = (y_val > 0.5).float()
                    
                    # Calculate IoU using proper tensor operations
                    intersection = (pred_binary * y_binary).sum()
                    union = (pred_binary + y_binary).sum() - intersection
                    val_iou += (intersection + 1e-6) / (union + 1e-6)

            avg_iou = val_iou / len(val_loader)
            scheduler.step(1 - avg_iou)

            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(model.state_dict(), "models/forgery_detector.pth")

            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val IoU: {avg_iou:.4f}")
    except Exception as e:
        print(f"Error in training forgery detector: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    print("Training transmission simulator...")
    train_transmission_simulator()
    print("\nTraining forgery detector...")
    train_forgery_detector()