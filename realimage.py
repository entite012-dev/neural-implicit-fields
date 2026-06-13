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
    """
    if not os.path.exists(image_path):
        # Agar physical image nahi milti toh testing ke liye ek dummy dummy matrix grid bana lena
        print(f"⚠️ Warning: '{image_path}' nahi mili! Testing ke liye dummy checkerboard matrix use ho rhi hai.")
        img_np = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        img_np[::32, ::32] = 1.0  # Simple spatial structure
    else:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_np = np.array(img, dtype=np.float32) / 255.0
        
    h, w, c = img_np.shape
    
    # Grid coordinates generate karna [-1, 1] range me
    x = np.linspace(-1, 1, w, dtype=np.float32)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Arrays ko continuous vectors format me stack aur flatten karna
    coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    targets = img_np.reshape(-1, 3)
    
    return torch.from_numpy(coords), torch.from_numpy(targets)

# APNI TARGET IMAGE KA PATH YAHAN SET KAREIN
# Agar path 'dog.png' hai toh wo automatically use ho jayegi, warna script dummy grid bana degi
IMAGE_PATH = "dog.png" 
coords, target_rgb = load_and_preprocess_image(IMAGE_PATH)


# --- STEP 2: SIREN (SINUSOIDAL REPRESENTATION NETWORK) ARCHITECTURE ---
class SirenLayer(nn.Module):
    """
    Individual SIREN Layer jo Custom Uniform Weight Initialization aur Periodic Sine use karti hai.
    """
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
        
    def init_weights(self):
        """
        SIREN ka custom mathematical uniform weight distribution initialization.
        """
        in_features = self.linear.in_features
        with torch.no_grad():
            if self.is_first:
                # First layer custom boundary initialization
                bound = 1 / in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers initialization code
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    Complete Continuous Implicit Neural Field Network
    """
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
    
    # Overfitting aur memorization control ke liye dynamic shuffling index matrix
    permutation = torch.randperm(total_pixels)
    
    for i in range(0, total_pixels, TRAIN_BATCH_SIZE):
        optimizer.zero_grad()
        
        # Dynamic GPU data streaming slices
        indices = permutation[i:i + TRAIN_BATCH_SIZE]
        batch_coords = coords[indices].to(device)
        batch_targets = target_rgb[indices].to(device)
        
        # Forward pass aur optimization step
        predictions = model(batch_coords)
        loss = criterion(predictions, batch_targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_coords.size(0)
        
    # Monitor Average Spatial Loss
    if epoch % 100 == 0 or epoch == 1:
        avg_loss = epoch_loss / total_pixels
        print(f"Epoch [{epoch:04d}/{EPOCHS}] -> Current Coordinate Convergence Loss: {avg_loss:.6f}")
        # Manual hardware cache dumping to prevent crashes
        gc.collect()
        torch.cuda.empty_cache()

print("\n🎯 Training fully completed without hardware throttling!")
print("=" * 60)


# --- STEP 4: 4K HIGH-RES VALIDATION EVALUATION (INFERENCE PHASE) ---
TEST_HIGH_RES = 2048  # Absolute 4K / Ultra-HD resolution test matrix grid (2048 x 2048)

print(f"🔍 Evaluating {TEST_HIGH_RES}x{TEST_HIGH_RES} implicit continuous mathematical field on RTX 3050...")

with torch.no_grad():
    model.eval()
    
    # Massive high-density grid optimization variables setup
    x_high = torch.linspace(-1, 1, TEST_HIGH_RES, device=device)
    y_high = torch.linspace(-1, 1, TEST_HIGH_RES, device=device)
    grid_x_h, grid_y_h = torch.meshgrid(x_high, y_high, indexing='ij')
    high_res_coords = torch.stack([grid_x_h, grid_y_h], dim=-1).view(-1, 2)
    
    # Inference phase ko bhi chote chunks me process karna taaki runtime evaluation par OOM na aaye
    EVAL_BATCH_SIZE = 131072 # 128k chunks for lightning fast evaluation without graphs
    predicted_high_list = []
    
    for i in range(0, high_res_coords.size(0), EVAL_BATCH_SIZE):
        chunk_coords = high_res_coords[i:i + EVAL_BATCH_SIZE]
        chunk_preds = model(chunk_coords)
        predicted_high_list.append(chunk_preds.cpu())
        
    # Array matrix reconstruction logic
    predicted_high_rgb = torch.cat(predicted_high_list, dim=0)
    hd_reconstructed = predicted_high_rgb.view(TEST_HIGH_RES, TEST_HIGH_RES, 3).numpy()
    hd_reconstructed = np.clip(hd_reconstructed, 0.0, 1.0)

# --- PIXEL PERFECT MATRIX SAVE (Bypassing Canvas Layout Interpolation Errors) ---
plt.imsave("dog_neural_4k.png", hd_reconstructed)
print("✅ Ultra-HD implicit representation asset successfully saved to 'dog_neural_4k.png' using plt.imsave!")


# --- APKA ORIGINAL PRESENTATION CANVAS CODE (Bina delete kiye safe rakha hai) ---
plt.figure(figsize=(12, 12))
plt.imshow(hd_reconstructed)
plt.axis('off')
plt.title(f"Implicit Continuous Neural Render - {TEST_HIGH_RES}x{TEST_HIGH_RES}", color='white', fontsize=14)
plt.gcf().patch.set_facecolor('black')
plt.savefig("dog_neural_4k_presentation.png", facecolor=plt.gcf().get_facecolor(), bbox_inches='tight', dpi=300)
print("🖼️ Human-readable presentation plot saved separately to 'dog_neural_4k_presentation.png'!")
print("=" * 60)


# --- STEP 5: BINARY DATA GENERATION FOR OUR GENERATIVE MANIFOLD DATASET ---
print("💾 Extracting optimized value vectors and wave attributes for dataset creation...")

extracted_snapshot = {
    "image_id": "dog_001",    # Dynamic processing setup token ID
    "prompt_token": 0,        # Meta identity tagging: 0 = Dog, 1 = Glasses
    "omega_0": OMEGA_0,       # Base signal scaling factor reference
    "layers": {}
}

with torch.no_grad():
    for name, param in model.named_parameters():
        # Multidimensional array keys mapping
        clean_name = name.replace(".", "_")
        extracted_snapshot["layers"][clean_name] = param.detach().cpu().numpy()

# Binary array payload payload save pipeline execution
output_filename = "extracted_siren_data_001.pt"
torch.save(extracted_snapshot, output_filename)

print(f"🔥 SUCCESS DUMP! Neural feature data stored perfectly in '{output_filename}'")
print("🚀 READY FOR THE HYPERNETWORK GENERATION EXPERIMENTS! NOW WE WIN FROM ALL SIDES!")
print("=" * 60)
