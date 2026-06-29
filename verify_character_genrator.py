import os
import torch
import torch.nn as nn
import numpy as np
import trimesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OMEGA_0 = 30.0
EVAL_BATCH_SIZE = 65536  # Safe VRAM batch size for training

# --- SIREN ARCHITECTURE ---
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega_0, np.sqrt(6 / in_features) / omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=4, out_features=1, omega_0=30.0):
        super().__init__()
        self.net = []
        self.net.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        self.final_linear = nn.Linear(hidden_features, out_features)
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        return self.final_linear(self.net(x))

# --- SAFE BATCHED SIGNED DISTANCE FIELD DATA GENERATOR ---
def generate_true_sdf_data(obj_path, num_samples=150000):
    print("📦 Loading mesh and generating True Signed Distance Fields (SDF)...")
    mesh = trimesh.load(obj_path, force='mesh')
    
    # 1. Surface points sample karo
    surf_points, _ = trimesh.sample.sample_surface(mesh, num_samples // 3)
    
    # Bounding box limits for empty space sampling
    v_min, v_max = mesh.bounds
    space_points_1 = surf_points + np.random.normal(0, 0.02, surf_points.shape)
    space_points_2 = np.random.uniform(v_min, v_max, (num_samples // 3, 3))
    
    all_points = np.vstack([surf_points, space_points_1, space_points_2])
    
    # 2. CHUNKED COMPUTATION (VS Code Crash Protection)
    print("⚡ Computing Signed Distances in safe batches to prevent RAM crash...")
    prox_query = trimesh.proximity.ProximityQuery(mesh)
    
    signed_distance_list = []
    chunk_size = 20000  # System RAM limit bracket
    
    for i in range(0, all_points.shape[0], chunk_size):
        chunk_points = all_points[i:i+chunk_size]
        chunk_sdf = prox_query.signed_distance(chunk_points)
        signed_distance_list.append(chunk_sdf)
        print(f"   Processed chunk {i // chunk_size + 1} / {int(np.ceil(all_points.shape[0] / chunk_size))}")
        
    signed_distance = np.concatenate(signed_distance_list, axis=0)
    
    # Standardize sign (Negative = Inside, Positive = Outside)
    signed_distance = -signed_distance 
    targets = signed_distance.reshape(-1, 1)
    
    # 3. Normalization inside [-1, 1] bounds for SIREN stability
    center = all_points.mean(axis=0)
    all_points -= center
    max_dist = np.max(np.sqrt(np.sum(all_points**2, axis=1)))
    coords = all_points / max_dist
    targets = targets / max_dist  
    
    return torch.tensor(coords, dtype=torch.float32).to(device), torch.tensor(targets, dtype=torch.float32).to(device)

if __name__ == "__main__":
    fbx_file_path = "data/body_block_fighter.obj" 
    coordinates, targets = generate_true_sdf_data(fbx_file_path, num_samples=150000)
    
    model = SIREN(in_features=3, hidden_features=256, hidden_layers=4, out_features=1, omega_0=OMEGA_0).to(device)
    print("\n🚀 SIREN True SDF Engine Instantiated.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    EPOCHS = 600
    total_points = coordinates.shape[0]
    
    model.train()
    for epoch in range(1, EPOCHS + 1):
        permutation = torch.randperm(total_points)
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, total_points, EVAL_BATCH_SIZE):
            optimizer.zero_grad()
            indices = permutation[i:i + EVAL_BATCH_SIZE]
            batch_x, batch_y = coordinates[indices], targets[indices]
            
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            
        if epoch == 1 or epoch % 50 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"🏅 Epoch [{epoch:04d}/{EPOCHS}] -> True SDF Loss (MSE): {avg_loss:.8f}")
            
    os.makedirs("serialized_assets", exist_ok=True)
    torch.save(model.state_dict(), "serialized_assets/fighter_geometry_implicit.pt")
    print("💾 True SDF Weights Saved Successfully!")