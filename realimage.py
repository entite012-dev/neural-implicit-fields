import os
import gc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. CUDA Memory Fragmentation aur OOM se bachne ke liye configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- HYPERPARAMETERS & CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 65536  # RTX 3050 ki 6GB VRAM ke liye sabse safe aur fast batch size
EPOCHS = 2000             # Structural mapping ke liye optimal epochs
OMEGA_0 = 30.0            # SIREN ka core frequency scaling parameter fine details ke liye

print(f"🚀 Execution Engine initialized on device: {device}")
print("=" * 60)

# --- STEP 1: IMAGE PROCESSING & COORDINATE MAPPING ---
def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """
    Image ko load karke normalized coordinates [-1, 1] aur target RGB vector me badalna.
    Axes inversion glitch ko fix karne ke liye indexing='xy' enforce kiya gaya hai.
    """
    if not os.path.exists(image_path):
        print(f"⚠️ Warning: '{image_path}' nahi mili! Testing ke liye dummy checkerboard matrix use ho rhi hai.")
        img_np = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        img_np[::32, ::32] = 1.0  
    else:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_np = np.array(img, dtype=np.float32) / 255.0
        
    h, w, c = img_np.shape
    
    # Grid coordinates generate karna [-1, 1] range me (Glitch Fix: indexing='xy')
    x = np.linspace(-1, 1, w, dtype=np.float32)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
    
    # Arrays ko continuous vectors format me stack aur flatten karna
    coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    targets = img_np.reshape(-1, 3)
    
    return torch.from_numpy(coords), torch.from_numpy(targets)


# ==============================================================================
# 🎯 DHYAN DEIN: AGAR DOG KA DATA NIKALNA HAI TOH YAHAN "dog.png" RAKHO, 
# AGAR GLASSES KA NIKALNA HAI TOH ISS LINE KO "glasses.png" KAR DENA.
# ==============================================================================
IMAGE_PATH = "dog.png"  
coords, target_rgb = load_and_preprocess_image(IMAGE_PATH)


# --- STEP 2: SIREN (SINUSOIDAL REPRESENTATION NETWORK) ARCHITECTURE ---
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
        
    def init_weights(self):
        in_features = self.linear.in_features
        with torch.no_grad():
            if self.is_first:
                bound = 1 / in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3, omega_0=30.0):
        super().__init__()
        self.net = []
        
        # Input Vector Projection Layer
        self.net.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        
        # Hidden Processing Continuous Layers Chain
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
            
        # Final Output Layer (Linear Projection to Physical RGB Values)
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        x = self.net(x)
        return self.final_linear(x)

# Model initialize karna
model = SIREN(omega_0=OMEGA_0).to(device)


# --- STEP 3: OPTIMIZED BATCHED TRAINING LOOP (THE HARDWARE FIX) ---
total_pixels = coords.size(0)
print(f"📊 Total Spatial Tokens to Process: {total_pixels} coordinates mapping tensors.")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print(f"🏋️ Training phase started for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    
    permutation = torch.randperm(total_pixels)
    
    for i in range(0, total_pixels, TRAIN_BATCH_SIZE):
        optimizer.zero_grad()
        
        indices = permutation[i:i + TRAIN_BATCH_SIZE]
        batch_coords = coords[indices].to(device)
        batch_targets = target_rgb[indices].to(device)
        
        predictions = model(batch_coords)
        loss = criterion(predictions, batch_targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_coords.size(0)
        
    if epoch % 100 == 0 or epoch == 1:
        avg_loss = epoch_loss / total_pixels
        print(f"Epoch [{epoch:04d}/{EPOCHS}] -> Current Coordinate Convergence Loss: {avg_loss:.6f}")
        gc.collect()
        torch.cuda.empty_cache()

print("\n🎯 Training fully completed without hardware throttling!")
print("=" * 60)


# --- STEP 4: 4K HIGH-RES VALIDATION EVALUATION (INFERENCE PHASE) ---
TEST_HIGH_RES = 2048  # Absolute 4K / Ultra-HD resolution matrix grid
print(f"🔍 Evaluating {TEST_HIGH_RES}x{TEST_HIGH_RES} implicit continuous mathematical field on RTX 3050...")

with torch.no_grad():
    model.eval()
    
    x_high = torch.linspace(-1, 1, TEST_HIGH_RES, device=device)
    y_high = torch.linspace(-1, 1, TEST_HIGH_RES, device=device)
    
    # Glitch Fix: Grid alignment ko strict Cartesian frames me rakhne ke liye indexing='xy'
    grid_x_h, grid_y_h = torch.meshgrid(x_high, y_high, indexing='xy')
    high_res_coords = torch.stack([grid_x_h, grid_y_h], dim=-1).view(-1, 2)
    
    EVAL_BATCH_SIZE = 131072 
    predicted_high_list = []
    
    for i in range(0, high_res_coords.size(0), EVAL_BATCH_SIZE):
        chunk_coords = high_res_coords[i:i + EVAL_BATCH_SIZE]
        chunk_preds = model(chunk_coords)
        predicted_high_list.append(chunk_preds.cpu())
        
    predicted_high_rgb = torch.cat(predicted_high_list, dim=0)
    hd_reconstructed = predicted_high_rgb.view(TEST_HIGH_RES, TEST_HIGH_RES, 3).numpy()
    hd_reconstructed = np.clip(hd_reconstructed, 0.0, 1.0)

# Render output save karna (Inversion fixed)
plt.imsave("neural_output_4k.png", hd_reconstructed)
print("✅ Ultra-HD implicit representation asset successfully saved to 'neural_output_4k.png'!")
print("=" * 60)


# ==============================================================================
# --- STEP 5: UPGRADED DATA EXTRACTION WITH SYSTEMATIC STRUCTURAL FLATTENING ---
# ==============================================================================
print("💾 Extracting optimized value vectors and wave attributes for dataset creation...")

# 1. Tumhara mathematical flattening block jo strict sequential order maintain karta hai
with torch.no_grad():
    ordered_vectors = []
    
    # State dict hamesha layers ke exact structural order me execute hota hai
    for name, param in model.state_dict().items():
        # Har ek weight aur bias layer ko ek strict order me flatten karke list me daalna
        ordered_vectors.append(param.detach().cpu().view(-1))
        
    # Pura model ab ek SINGLE LONG ROW VECTOR ban chuka hai! (Strict Positional Alignment Locked)
    flat_model_row = torch.cat(ordered_vectors).numpy()

# ==============================================================================
# 🎯 DHYAN DEIN: AGAR DOG KA DATA HAI TOH PROFILE TOKENS CHANGE KAREIN:
# Dog ke liye:       "image_id": "dog_001",     "prompt_token": 0,  "output_filename": "extracted_siren_data_001.pt"
# Glasses ke liye:   "image_id": "glasses_001", "prompt_token": 1,  "output_filename": "extracted_siren_data_002.pt"
# ==============================================================================
extracted_snapshot = {
    "image_id": "dog_001",                 # Agar glasses ho toh change to "glasses_001"
    "prompt_token": 0,                     # Dog = 0, Glasses = 1
    "omega_0": OMEGA_0,                    # Base frequency scaling factor
    "flat_weights_vector": flat_model_row   # Aligned ordered array vector 
}

output_filename = "extracted_siren_data_001.pt"  # Glasses ke liye change to "extracted_siren_data_002.pt"
torch.save(extracted_snapshot, output_filename)

print(f"🔥 SUCCESS DUMP! Ordered data vector of shape {flat_model_row.shape} stored perfectly in '{output_filename}'")
print("🚀 READY FOR THE HYPERNETWORK GENERATION EXPERIMENTS! POSITION ALIGNMENT LOCKED!")
print("=" * 60)
