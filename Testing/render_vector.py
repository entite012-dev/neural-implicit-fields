import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- ENGINE ENGINE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OMEGA_0 = 30.0
TEST_HIGH_RES = 2048 # Absolute 4K Rendering Grid

print(f"🎬 Decoding Vector Engine running on: {device}")
print("=" * 60)

# --- STEP 1: LOAD HYBRID PARAMETER PAYLOAD ---
HYBRID_PATH = "hybrid_composition_vector.pt"
if not os.path.exists(HYBRID_PATH):
    raise FileNotFoundError(f"⚠️ Error: '{HYBRID_PATH}' nahi mili! Pehle hypergenerator.py chalayein.")

print(f"💾 Loading generated parameter matrix from '{HYBRID_PATH}'...")
payload = torch.load(HYBRID_PATH, weights_only=False)
flat_weights = payload["synthesized_weights"]
print(f"✅ Loaded parameter array vector of shape: {flat_weights.shape}")


# --- STEP 2: RECONSTRUCT SIREN GRAPH TO RE-INJECT WEIGHTS ---
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3, omega_0=30.0):
        super().__init__()
        self.net = []
        self.net.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        self.final_linear = nn.Linear(hidden_features, out_features)
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        x = self.net(x)
        return self.final_linear(x)

# Model initialization
model = SIREN(omega_0=OMEGA_0).to(device)


# --- STEP 3: THE INJECTION JUGAD (SPLITTING FLAT VECTOR BACK TO LAYERS) ---
print("💉 Injecting synthetic parameter blocks back into neural network layers...")
state_dict = model.state_dict()
current_pointer = 0

with torch.no_grad():
    for name, param in state_dict.items():
        # layer weight ya bias ka exact total elements size nikalna
        num_elements = param.numel()
        
        # Long flat vector se exact utna piece slice (cut) karna
        layer_slice = flat_weights[current_pointer : current_pointer + num_elements]
        
        # Us sliced array ko layer ke original dimension shape me reshape karna
        reshaped_tensor = torch.from_numpy(layer_slice).view(param.shape).to(device)
        
        # State dict ke andar parameters override kar dena
        param.copy_(reshaped_tensor)
        
        # Pointer ko agle block par shift karna
        current_pointer += num_elements

print("🔒 Structural weight layers fully populated and locked!")
print("=" * 60)


# --- STEP 4: GENERATE 4K SPATIAL COORDINATE REPRESENTATION ---
print(f"🔍 Interpolating analytical field at extreme {TEST_HIGH_RES}x{TEST_HIGH_RES} limits...")

with torch.no_grad():
    model.eval()
    
    # Strict 'xy' Cartesian indexing meshgrid execution
    x_high = np.linspace(-1, 1, TEST_HIGH_RES, dtype=np.float32)
    y_high = np.linspace(-1, 1, TEST_HIGH_RES, dtype=np.float32)
    grid_x_h, grid_y_h = np.meshgrid(x_high, y_high, indexing='xy')
    
    high_res_coords_np = np.stack([grid_x_h, grid_y_h], axis=-1).reshape(-1, 2)
    high_res_coords = torch.from_numpy(high_res_coords_np).to(device)
    
    EVAL_BATCH_SIZE = 131072 
    predicted_high_list = []
    
    for i in range(0, high_res_coords.size(0), EVAL_BATCH_SIZE):
        chunk_coords = high_res_coords[i:i + EVAL_BATCH_SIZE]
        chunk_preds = model(chunk_coords)
        predicted_high_list.append(chunk_preds.cpu())
        
    predicted_high_rgb = torch.cat(predicted_high_list, dim=0)
    
    # Matrix reconstruction (H, W, Channels)
    hd_reconstructed = predicted_high_rgb.view(TEST_HIGH_RES, TEST_HIGH_RES, 3).numpy()
    hd_reconstructed = np.clip(hd_reconstructed, 0.0, 1.0)

# Final Render Plot Save
OUTPUT_IMG_NAME = "final_hybrid_composition_4k.png"
plt.imsave(OUTPUT_IMG_NAME, hd_reconstructed)

print(f"🎉 BOOM! Your continuous blended image successfully saved to '{OUTPUT_IMG_NAME}'")
print("🔥 FULL END-TO-END RE-INJECTION PIPELINE EXECUTED SUCCESSFULLY!")
print("=" * 60)