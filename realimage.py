import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Connect with your RTX 3050 CUDA cores
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Accelerating Photorealistic Engine on: {device}")

# ==============================================================================
# STEP 1: THE SIREN LAYER ENGINE (Sinusoidal Representation Network)
# ==============================================================================
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
        
    def init_weights(self):
        # Sabhi in-place initializations ko no_grad ke andar rakhna zaroori hai
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.w0, 
                                            np.sqrt(6 / self.linear.in_features) / self.w0)
            
            # FIX: Isko block ke andar laya taaki leaf variable error na aaye
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        # Applying core trigonometric function over coordinate inputs
        return torch.sin(self.w0 * self.linear(x))

class SirenMLP(nn.Module):
    def __init__(self, hidden_features=256, hidden_layers=4, w0=30.0):
        super().__init__()
        layers = [SirenLayer(2, hidden_features, w0=w0, is_first=True)]
        
        for _ in range(hidden_layers):
            layers.append(SirenLayer(hidden_features, hidden_features, w0=w0))
            
        # Final Layer outputs RGB coordinates directly
        final_linear = nn.Linear(hidden_features, 3)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / w0, np.sqrt(6 / hidden_features) / w0)
            final_linear.bias.zero_()
            
        self.network = nn.Sequential(*layers, final_linear)

    def forward(self, coords):
        return self.network(coords)

# ==============================================================================
# STEP 2: PRE-PROCESSING DATA ENGINE
# ==============================================================================
def load_target_image(image_path, size=256):
    # Load your dog photo or any reference image
    img = cv2.imread(image_path)
    if img is None:
        # Fallback dummy grid if file doesn't exist yet
        print("⚠️ Reference photo not found! Creating a high-gradient matrix pattern.")
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(dummy, (size//2, size//2), size//3, (200, 150, 100), -1) # Dog face spot
        cv2.circle(dummy, (size//2 - 40, size//2 - 40), 15, (20, 20, 20), -1) # Eye
        cv2.circle(dummy, (size//2 + 40, size//2 - 40), 15, (20, 20, 20), -1) # Eye
        return torch.tensor(dummy, dtype=torch.float32, device=device) / 255.0
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return torch.tensor(img, dtype=torch.float32, device=device) / 255.0

def get_coordinate_grid(size=256):
    # Normalized Spatial Space (-1 to 1)
    x = torch.linspace(-1, 1, size, device=device)
    y = torch.linspace(-1, 1, size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
    return coords

# ==============================================================================
# STEP 3: THE TRAINING / LEARNING LOOP
# ==============================================================================
GRID_SIZE = 512
target_img = load_target_image("dog_photo.jpg", size=GRID_SIZE)
flat_target = target_img.view(-1, 3)
coords_grid = get_coordinate_grid(size=GRID_SIZE)

# Deploy SIREN Network Architecture
model = SirenMLP(hidden_features=512, hidden_layers=4, w0=30.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("\n🚀 Model Training Initiated. Learning image features...")
plt.ion() # Interactive mode for realtime plotting

for epoch in range(50001):
    optimizer.zero_grad()
    
    # Model predicts color values for each coordinate mapping
    predicted_rgb = model(coords_grid)
    loss = criterion(predicted_rgb, flat_target)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss (MSE): {loss.item():.6f}")
        
        # Real-time reconstruction lookup
        reconstructed = predicted_rgb.view(GRID_SIZE, GRID_SIZE, 3).detach().cpu().numpy()
        reconstructed = np.clip(reconstructed, 0.0, 1.0)
        
        plt.imshow(reconstructed)
        plt.title(f"Neural Reconstruction Epoch: {epoch}")
        plt.axis('off')
        plt.pause(0.1)

plt.ioff()
print("Learning Engine execution completed!")
plt.show()

# ==============================================================================
# STEP 4: INFINITE RESOLUTION SUPER-SAMPLING TEST
# ==============================================================================
print("\n" + "=" * 60)
print("🚀 LAUNCHING ULTRA-HD IMPLICIT RECONSTRUCTION (SUPER-SAMPLING)")
print("=" * 60)

# 4K / Ultra-HD resolution test matrix grid (2048 x 2048)
TEST_HIGH_RES = 2048 

with torch.no_grad():
    # Creating a massive high-density coordinate space grid on RTX 3050
    x_high = torch.linspace(-1, 1, TEST_HIGH_RES, device=device)
    y_high = torch.linspace(-1, 1, TEST_HIGH_RES, device=device)
    grid_x_h, grid_y_h = torch.meshgrid(x_high, y_high, indexing='ij')
    high_res_coords = torch.stack([grid_x_h, grid_y_h], dim=-1).view(-1, 2)
    
 # Querying the learned SIREN weights for the massive grid
print(f"Evaluating {TEST_HIGH_RES}x{TEST_HIGH_RES} implicit continuous mathematical field...")
predicted_high_rgb = model(high_res_coords)

# Reshaping back to image space
hd_reconstructed = predicted_high_rgb.view(TEST_HIGH_RES, TEST_HIGH_RES, 3).cpu().numpy()
hd_reconstructed = np.clip(hd_reconstructed, 0.0, 1.0)

# ==================== NAYA AUR EXACT CODE ====================
# Matplotlib ka use karke bina padding aur title ke exact pixels save karna
plt.imsave("dog_neural_4k.png", hd_reconstructed)
# =============================================================

print(" Ultra-HD implicit representation asset successfully saved to 'dog_neural_4k.png'!")
print("=" * 60)
