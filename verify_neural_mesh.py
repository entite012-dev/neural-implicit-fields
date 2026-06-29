import os
import torch
import torch.nn as nn
import numpy as np
import trimesh

# --- CONFIGURATIONS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OMEGA_0 = 30.0

# --- SIREN ARCHITECTURE (Must match training exactly) ---
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=4, out_features=3, omega_0=30.0):
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

# --- VALIDATION ENGINE ---
def run_geometry_checker(obj_path, weights_path):
    print("🔍 Starting Neural Geometry Validation Checker...")
    print("=" * 60)
    
    # 1. Ground Truth Data load karo (Original Mesh)
    if not os.path.exists(obj_path):
        print(f"❌ Error: Ground truth mesh '{obj_path}' nahi mila.")
        return
    
    mesh = trimesh.load(obj_path, force='mesh')
    sampled_points, _ = trimesh.sample.sample_surface(mesh, 50000) # Test on 50k points
    
    # Normalize (Same as training)
    center = sampled_points.mean(axis=0)
    sampled_points -= center
    max_dist = np.max(np.sqrt(np.sum(sampled_points**2, axis=1)))
    gt_coords = sampled_points / max_dist
    gt_targets = (gt_coords + 1.0) / 2.0
    
    # Tensors mein convert karo
    X_test = torch.tensor(gt_coords, dtype=torch.float32).to(device)
    Y_true = torch.tensor(gt_targets, dtype=torch.float32).to(device)
    
    # 2. Model Load Karo aur Weights Inject Karo
    model = SIREN(in_features=3, hidden_features=512, hidden_layers=4, out_features=3, omega_0=OMEGA_0).to(device)
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: Trained weights file '{weights_path}' nahi mili!")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✅ Trained Neural Weights Successfully Loaded into 512-MLP Model.")
    
    # 3. Predict Neural Space Geometry
    with torch.no_grad():
        Y_pred = model(X_test)
        
    # 4. MATHS CHECK: Metrics Calculation for Disturbance & Noise
    # Mean Absolute Error (MAE) check karta hai ki average bhatkao kitna hai
    mae_loss = torch.mean(torch.abs(Y_pred - Y_true)).item()
    # Max Deviation check karta hai ki poori body mein sabse bada glitch/disturbance kahan hai
    max_deviation = torch.max(torch.abs(Y_pred - Y_true)).item()
    
    print("-" * 60)
    print("📊 NEURAL INTEGRITY REPORT:")
    print(f"🔹 Mean Shape Distortion (MAE): {mae_loss:.6f}")
    print(f"🔹 Maximum Coordinate Glitch:   {max_deviation:.6f}")
    print("-" * 60)
    
    # 5. FINAL PASS/FAIL CRITERIA
    if mae_loss < 0.005 and max_deviation < 0.05:
        print("🎉 PASS: Neural Representation is 100% stable! No major disturbance detected.")
        print("💡 X Bot's continuous coordinate space is perfectly preserved inside the model.")
    elif mae_loss < 0.02:
        print("⚠️ WARNING: Mesh is mostly fine, but slight blurriness/disturbance detected in tiny details.")
    else:
        print("❌ FAIL: High disturbance detected! Model parameters are distorted or underfitted.")

if __name__ == "__main__":
    # Apne paths check kar lena
    mesh_file = "data/body_block_fighter.obj"
    trained_weights = "serialized_assets/fighter_geometry_implicit.pt"
    
    run_geometry_checker(mesh_file, trained_weights)