import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 🛠️ CHANGES APPLIED: UPPER LAYER HIGH-CAPACITY HYPERPARAMETERS
# ==============================================================================
TRAIN_BATCH_SIZE = 65536
EPOCHS = 4000       # Network bada hai toh optimization ko 4000 epochs diye hain
OMEGA_0 = 30.0
LATENT_DIM = 32     # 4 se badha kar 32 kiya -> More space for micro-features
HIDDEN_DIM = 512    # 256 se badha kar 512 neurons kiye -> High capacity representation

print(f"🚀 Upgraded Latent Conditioning Core running on: {device}")
print(f"🧠 Network Width: {HIDDEN_DIM} neurons | Latent Space: {LATENT_DIM} dimensions")
print("=" * 60)

# --- FUNCTION: LOAD IMAGE AS TENSOR ---
def get_image_data(path, target_size=(512, 512)):
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ '{path}' nahi mili! Apne folder me 'dog.png' aur 'glasses.png' check karein.")
    img = Image.open(path).convert('RGB').resize(target_size)
    img_np = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_np.reshape(-1, 3))

# Target matrices load karna
target_dog = get_image_data("dog.png")
target_glasses = get_image_data("glasses.png")

# Unified grid generation (indexing='xy' dynamic fix)
x = np.linspace(-1, 1, 512, dtype=np.float32)
y = np.linspace(-1, 1, 512, dtype=np.float32)
grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
base_coords = torch.from_numpy(np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2))


# ==============================================================================
# 🛠️ CHANGES APPLIED: 32-DIMENSIONAL DETERMINISTIC EMBEDDINGS
# ==============================================================================
# Dog ke liye pehla element 1 baaki sab 0
z_dog = torch.zeros(LATENT_DIM)
z_dog[0] = 1.0  

# Glasses ke liye dusra element 1 baaki sab 0
z_glasses = torch.zeros(LATENT_DIM)
z_glasses[1] = 1.0  

print("📊 High-dimensional data profiles loaded successfully.")


# --- STEP 2: CONDITIONED SIREN MODEL ARCHITECTURE ---
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

class ConditionedSIREN(nn.Module):
    def __init__(self, in_features=2, latent_dim=32, hidden_features=512, hidden_layers=4, out_features=3, omega_0=30.0):
        super().__init__()
        self.net = []
        
        # Pheli layer ab XY coordinates + 32D Latent Vector dono legi!
        self.net.append(SirenLayer(in_features + latent_dim, hidden_features, is_first=True, omega_0=omega_0))
        
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
            
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords, latent):
        # Latent vector ko replicate karke coordinates ke sath attach karna
        x = torch.cat([coords, latent], dim=-1)
        x = self.net(x)
        return self.final_linear(x)

# Model initialization using high capacity config
model = ConditionedSIREN(latent_dim=LATENT_DIM, hidden_features=HIDDEN_DIM, omega_0=OMEGA_0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()


# --- STEP 3: THE COMPOSITE JOINT TRAINING LOOP ---
print(f"🏋️ High-Capacity multi-identity training started for {EPOCHS} epochs...")
total_pixels = base_coords.size(0)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    
    permutation = torch.randperm(total_pixels)
    
    for i in range(0, total_pixels, TRAIN_BATCH_SIZE):
        indices = permutation[i:i + TRAIN_BATCH_SIZE]
        b_coords = base_coords[indices].to(device)
        
        # --- Batch 1: Dog Matrix ---
        optimizer.zero_grad()
        b_latent_dog = z_dog.repeat(b_coords.size(0), 1).to(device)
        b_targets_dog = target_dog[indices].to(device)
        preds_dog = model(b_coords, b_latent_dog)
        loss_dog = criterion(preds_dog, b_targets_dog)
        loss_dog.backward()
        optimizer.step()
        
        # --- Batch 2: Glasses Matrix ---
        optimizer.zero_grad()
        b_latent_glass = z_glasses.repeat(b_coords.size(0), 1).to(device)
        b_targets_glass = target_glasses[indices].to(device)
        preds_glass = model(b_coords, b_latent_glass)
        loss_glass = criterion(preds_glass, b_targets_glass)
        loss_glass.backward()
        optimizer.step()
        
        epoch_loss += (loss_dog.item() + loss_glass.item()) * b_coords.size(0)
        
    if epoch % 400 == 0 or epoch == 1:
        print(f"Epoch [{epoch:04d}/{EPOCHS}] -> Total Multimodal Loss: {epoch_loss / (total_pixels * 2):.6f}")

print("🎯 High-capacity shared manifold optimization successfully converged!")
print("=" * 60)


# --- STEP 4: GENERATING THE SHARP INTERPOLATION BLEND ---
TEST_RES = 1024 
print(f"🔮 Rendering Sharp Space Interpolation at {TEST_RES}x{TEST_RES}...")

x_high = np.linspace(-1, 1, TEST_RES, dtype=np.float32)
y_high = np.linspace(-1, 1, TEST_RES, dtype=np.float32)
grid_x_h, grid_y_h = np.meshgrid(x_high, y_high, indexing='xy')
high_res_coords = torch.from_numpy(np.stack([grid_x_h, grid_y_h], axis=-1).reshape(-1, 2)).to(device)

# 🛠️ CHANGES APPLIED: Creating a balanced 32-dimensional blended latent code!
z_blend = torch.zeros(LATENT_DIM)
z_blend[0] = 0.5  # 50% Dog features
z_blend[1] = 0.5  # 50% Glasses features
z_blend = z_blend.repeat(high_res_coords.size(0), 1).to(device)

model.eval()
with torch.no_grad():
    EVAL_BATCH_SIZE = 131072
    predicted_list = []
    
    for i in range(0, high_res_coords.size(0), EVAL_BATCH_SIZE):
        c_chunk = high_res_coords[i:i + EVAL_BATCH_SIZE]
        l_chunk = z_blend[i:i + EVAL_BATCH_SIZE]
        preds = model(c_chunk, l_chunk)
        predicted_list.append(preds.cpu())
        
    final_rgb = torch.cat(predicted_list, dim=0).view(TEST_RES, TEST_RES, 3).numpy()
    final_rgb = np.clip(final_rgb, 0.0, 1.0)

# Saving the continuous sharp master stroke output image
plt.imsave("latent_hybrid_blend_1k.png", final_rgb)
print("🎉 SUCCESS! High-capacity rendering saved to 'latent_hybrid_blend_1k.png'!")
print("=" * 60)
