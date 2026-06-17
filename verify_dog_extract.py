import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OMEGA_0 = 30.0
TEST_RES = 512  # Verification ke liye standard resolution kaafi hai

print(f"🕵️ Target Verification Engine Running On: {device}")
print("=" * 60)

# --- STEP 1: LOAD SERIALLY EXTRACTED ASSET ---
DOG_PT_PATH = "extracted_siren_data_001.pt"

if not os.path.exists(DOG_PT_PATH):
    raise FileNotFoundError(f"⚠️ '{DOG_PT_PATH}' nahi mili! Pehle realimage.py chala kar dog ka data extract karein.")

print(f"💾 Loading extracted snapshot from '{DOG_PT_PATH}'...")
payload = torch.load(DOG_PT_PATH, weights_only=False)
flat_weights = payload["flat_weights_vector"]
print(f"📦 Extracted Flat Vector Shape: {flat_weights.shape}")


# --- STEP 2: RECONSTRUCT THE STANDALONE SIREN GRAPH ---
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

# Baseline layout setup
model = SIREN(omega_0=OMEGA_0).to(device)


# --- STEP 3: WEIGHT INJECTION ROUTINE ---
print("💉 Re-assembling flat parameter blocks back into SIREN model...")
state_dict = model.state_dict()
current_pointer = 0

with torch.no_grad():
    for name, param in state_dict.items():
        num_elements = param.numel()
        
        # Exact chunk extraction
        layer_slice = flat_weights[current_pointer : current_pointer + num_elements]
        
        # Reshape and device assignment
        reshaped_tensor = torch.from_numpy(layer_slice).view(param.shape).to(device)
        
        # Loading back into the model weights structure
        param.copy_(reshaped_tensor)
        current_pointer += num_elements

print("🔒 Weights successfully injected and locked into layers.")
print("=" * 60)


# --- STEP 4: COORDINATE EVALUATION ---
print(f"🔍 Generating verification image rendering matrix ({TEST_RES}x{TEST_RES})...")

with torch.no_grad():
    model.eval()
    
    # Grid coordinate structure replication
    x = np.linspace(-1, 1, TEST_RES, dtype=np.float32)
    y = np.linspace(-1, 1, TEST_RES, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
    
    coords_np = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    coords_tensor = torch.from_numpy(coords_np).to(device)
    
    EVAL_BATCH_SIZE = 65536
    predicted_list = []
    
    for i in range(0, coords_tensor.size(0), EVAL_BATCH_SIZE):
        chunk_coords = coords_tensor[i:i + EVAL_BATCH_SIZE]
        chunk_preds = model(chunk_coords)
        predicted_list.append(chunk_preds.cpu())
        
    final_rgb = torch.cat(predicted_list, dim=0).view(TEST_RES, TEST_RES, 3).numpy()
    final_rgb = np.clip(final_rgb, 0.0, 1.0)

# Render Output Save
plt.imsave("dog_reconstructed_verify.png", final_rgb)
print("🎯 DIAGNOSTICS COMPLETE!")
print("💾 Result saved to 'dog_reconstructed_verify.png'")
print("=" * 60)