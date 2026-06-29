import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
from skimage import measure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OMEGA_0 = 30.0

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
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

def render_solid_mesh_sdf(weights_path, output_obj_path, grid_resolution=160):
    print(f"🏗️ Initializing Custom Triangle/Box Mask Engine (Resolution: {grid_resolution})...")
    print("=" * 60)
    
    model = SIREN(in_features=3, hidden_features=256, hidden_layers=4, out_features=1, omega_0=OMEGA_0).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Grid parameters jahan character full capture ho sake
    x = np.linspace(-0.85, 0.85, grid_resolution)
    y = np.linspace(-0.85, 0.75, grid_resolution) 
    z = np.linspace(-0.65, 0.65, grid_resolution)
    
    grid_X, grid_Y, grid_Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([grid_X.ravel(), grid_Y.ravel(), grid_Z.ravel()], axis=1)
    
    X_query = torch.tensor(grid_points, dtype=torch.float32).to(device)
    distances = []
    batch_size = 131072
    
    with torch.no_grad():
        for i in range(0, X_query.shape[0], batch_size):
            batch_pred = model(X_query[i:i+batch_size])
            distances.append(batch_pred.cpu().numpy())
            
    volume = np.concatenate(distances, axis=0).reshape(grid_resolution, grid_resolution, grid_resolution)

    print("📐 Slicing zero surface layer...")
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
    verts += np.array([x[0], y[0], z[0]])

    # --- 🔥 AAPKA TRIANGLE/BOX CUTTER MASK 🔥 ---
    # Hum har axis ko uski perfect limit par trim karenge taaki T-Pose haath bilkul na katyein!
    print("✂️ Applying sharp coordinate cutter to protect hands and clear noise...")
    
    # X (haath se haath) ko poora khula rakha (-0.8 se 0.8), par Z (aage-peeche ka motapa) tight kar diya (-0.3 se 0.3)
    valid_mask = (verts[:, 0] > -0.8) & (verts[:, 0] < 0.8) & \
                 (verts[:, 1] > -0.85) & (verts[:, 1] < 0.75) & \
                 (verts[:, 2] > -0.30) & (verts[:, 2] < 0.30)
    
    # Re-indexing geometry to drop masked artifacts
    unique_indices = np.where(valid_mask)[0]
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
    
    filtered_faces = []
    for face in faces:
        if valid_mask[face[0]] and valid_mask[face[1]] and valid_mask[face[2]]:
            filtered_faces.append([index_mapping[face[0]], index_mapping[face[1]], index_mapping[face[2]]])
            
    clean_verts = verts[valid_mask]
    clean_faces = np.array(filtered_faces)
    clean_normals = normals[valid_mask]

    print("🧹 Cleaning isolated components to ensure zero background noise...")
    raw_mesh = trimesh.Trimesh(vertices=clean_verts, faces=clean_faces, vertex_normals=clean_normals)
    
    # Split standalone parts if any left
    components = raw_mesh.split(only_watertight=False)
    if len(components) > 1:
        solid_mesh = max(components, key=lambda m: len(m.vertices))
    else:
        solid_mesh = raw_mesh

    # Niche se solid watertight block banane ke liye hole-filling
    if not solid_mesh.is_watertight:
        solid_mesh.fill_holes()

    print(f"💾 Saving complete clean solid mesh...")
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    solid_mesh.export(output_obj_path)
    print(f"🎉 SUCCESS! TRIANGLE-BOUND CUT COMPLETE: {output_obj_path}")

if __name__ == "__main__":
    render_solid_mesh_sdf("serialized_assets/fighter_geometry_implicit.pt", "outputs/solid_neural_x_bot.obj")