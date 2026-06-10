import os
import csv
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Connect with CUDA Accelerator (RTX 3050)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Extraction Engine active on: {device}")

# ==============================================================================
# 🧠 RE-INITIALIZE THE EXTRACTOR NETWORK
# ==============================================================================
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Stream 1: Predict geometry vectors
        self.vector_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3), # Outputting 3 main spatial parameters
            nn.Sigmoid()
        )
        # Stream 2: Predict sinusoidal high frequency texture waves
        self.wave_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        geo_vectors = self.vector_head(features)
        wave_freq = self.wave_head(features) * 100.0 # scale wave frequency max to 100Hz
        return geo_vectors, wave_freq

# Initialize model and push to GPU
model = ImageFeatureExtractor().to(device)
model.eval()

# Standard PyTorch normalization matrix for vision tasks
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# 📊 CSV WRITER PIPELINE
# ==============================================================================
csv_file_path = "dataset.csv"

# Define columns headers for the matrix dataset table
headers = ["image_name", "predicted_label", "head_radius", "ear_spacing", "ear_height", "fur_density_wave"]

print(f"📝 Initializing structured sheet file: {csv_file_path}")
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers) # Write table titles row

    # --- PROCESS REAL PHOTOS AND SAVE ---
    # Put all your animal pictures inside your project folder or subfolder
    image_files = [img for img in os.listdir('.') if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("⚠️ No images found in the current directory! Put some cat/mouse pictures here.")
    else:
        print(f"📂 Found {len(image_files)} assets. Simulating batch conversion to matrices...")
        
        for idx, img_name in enumerate(image_files):
            try:
                # 1. Image context analysis
                img_path = os.path.join('.', img_name)
                img = Image.open(img_path).convert('RGB')
                tensor_img = transform(img).unsqueeze(0).to(device)
                
                # 2. Extract vectors and wave frequencies via PyTorch tensor matrix
                with torch.no_grad():
                    geo_vecs, wave_freq = model(tensor_img)
                
                vec_data = geo_vecs.cpu().numpy()[0]
                freq_val = wave_freq.cpu().item()
                
                # 3. Label classifier logic based on file names (e.g. cat_1.jpg, mouse_2.jpg)
                label = "unknown"
                if "cat" in img_name.lower(): label = "cat"
                elif "mouse" in img_name.lower(): label = "mouse"
                elif "bear" in img_name.lower(): label = "bear"
                elif "dog" in img_name.lower(): label = "dog"
                
                # Scale values to clean bounds
                hr = float(vec_data[0] * 0.3 + 0.2)
                es = float(vec_data[1] * 0.2 + 0.15)
                eh = float(vec_data[2] * 0.2 + 0.2)
                
                # 4. Append row directly to CSV row buffer stream
                writer.writerow([img_name, label, f"{hr:.4f}", f"{es:.4f}", f"{eh:.4f}", f"{freq_val:.4f}"])
                print(f" └── [{idx+1}/{len(image_files)}] Parsed {img_name} -> Saved to Row Matrix.")
                
            except Exception as e:
                print(f" ❌ Failed parsing {img_name}: {str(e)}")

print(f"\n🎉 Process Complete! Dataset saved successfully inside: '{csv_file_path}'")