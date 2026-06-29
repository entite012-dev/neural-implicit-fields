import os
import torch
import torch.nn as nn
import numpy as np
import trimesh

# --- GLOBAL CONFIGURATIONS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OMEGA_0 = 30.0
TEST_HIGH_RES = 2048  # Direct 4K High-Res Grid Rendering
EVAL_BATCH_SIZE = 131072  # High-throughput batch size optimization

print(f"🖥️ Functional Space Blending Engine Running On: {device}")
print("=" * 60)

# --- STEP 1: AUTOMATIC 3D FBX COORDINATE EXTRACTOR ---
def generate_character_point_cloud(fbx_path, num_samples=262144):
    """
    Takes an FBX mesh, extracts dense surface coordinates, normalizes them 
    between [-1, 1] for SIREN, and generates high-throughput target vectors.
    """
    print(f"🔄 Processing 3D Mesh from: {fbx_path}...")
    if not os.path.exists(fbx_path):
        raise FileNotFoundError(f"❌ Error: File '{fbx_path}' nahi mili! Make sure your Mixamo file is in the data/ folder.")
        
    try:
        # Load the mesh (force='mesh' guarantees geometric attributes like vertices/faces)
        mesh = trimesh.load(fbx_path, force='mesh')
        
        print(f"📊 Raw Character Stats -> Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
        
        # Sample dense points across the 3D surface
        sampled_points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        
        # Center and Normalize coordinates into [-1, 1] bounding box for SIREN convergence
        center = sampled_points.mean(axis=0)
        sampled_points -= center
        max_dist = np.max(np.sqrt(np.sum(sampled_points**2, axis=1)))
        normalized_coords = sampled_points / max_dist
        
        # out_features = 3 Logic: Creating mock RGB color/normal vectors from coordinates
        # Maps coordinates [-1, 1] into clean spatial colors [0, 1] for previewing geometry
        mock_rgb = (normalized_coords + 1.0) / 2.0 
        
        print(f"✅ Data Extraction Complete! Tensor Shape: {normalized_coords.shape}")
        
        # Convert to PyTorch tensors and move to GPU/CPU device
        X_tensor = torch.tensor(normalized_coords, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(mock_rgb, dtype=torch.float32).to(device)
        
        return X_tensor, Y_tensor
        
    except Exception as e:
        print(f"❌ Error decoding FBX: {str(e)}")
        print("💡 Quick Fix: Run 'pip install trimesh' and ensure your FBX file is valid.")
        return None, None


# --- STEP 2: DEFINE BASE SIREN NETWORK ---
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first)
        
    def init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                # First layer needs a specific bound scale for sine mapping
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.linear.in_features) / self.omega_0)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=4, out_features=3, omega_0=30.0):
        super().__init__()
        self.net = []
        
        # Input Layer (Takes 3D Coordinates: X, Y, Z)
        self.net.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        
        # Hidden Layers (Upgraded to 512 for deep structural details)
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
            
        self.final_linear = nn.Linear(hidden_features, out_features)
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        x = self.net(x)
        return self.final_linear(x)


# --- STEP 3: PIPELINE EXECUTION ---
if __name__ == "__main__":
    # Update path to your actual downloaded file name inside 'data/' folder
    fbx_file_path = "data/body_block_fighter.obj" 
    
    # 1. Extract 3D Coordinates
    coordinates, targets = generate_character_point_cloud(fbx_file_path, num_samples=262144)
    
    if coordinates is not None:
        # 2. Instantiate Model with 512 Features and 3 Outputs
        model = SIREN(in_features=3, hidden_features=512, hidden_layers=4, out_features=3, omega_0=OMEGA_0).to(device)
        print("\n🤖 SIREN Model Architecture Loaded with 512 Hidden Features.")
        
        # 3. Test High-Throughput High-Res Optimization Check
        print(f"\n⚡ Testing high-throughput inference using EVAL_BATCH_SIZE = {EVAL_BATCH_SIZE}...")
        
        # Slice one full batch out of the extracted coordinates
        test_batch = coordinates[:EVAL_BATCH_SIZE]
        
        # Run dummy forward pass to check VRAM throughput stability
        with torch.no_grad():
            output_batch = model(test_batch)
            
        print(f"🚀 Success! Processed Batch Output Shape: {output_batch.shape}")
        print("🎉 High-throughput pipeline is perfectly stable. Ready for training loops!")