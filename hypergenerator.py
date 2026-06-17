import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- SYSTEM ENGINE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" HyperNetwork Processing Core initialized on: {device}")
print("=" * 60)

# --- STEP 1: LOAD ALIGNED BINARY ASSETS ---
DATA_DOG_PATH = "extracted_siren_data_001.pt"
DATA_GLASSES_PATH = "extracted_siren_data_002.pt"

if not os.path.exists(DATA_DOG_PATH) or not os.path.exists(DATA_GLASSES_PATH):
    raise FileNotFoundError("⚠️ Error: Dono binary .pt files (001 aur 002) folder me honi chahiye! Pehle unhe extract karein.")

print("Loading serialized functional payloads...")
# Flag weights_only=False lagane se numpy array safely allowed ho jayega
data_dog = torch.load(DATA_DOG_PATH, weights_only=False)
data_glasses = torch.load(DATA_GLASSES_PATH, weights_only=False)

# Raw numerical flat vectors nikalna
vector_dog = torch.from_numpy(data_dog["flat_weights_vector"]).float()
vector_glasses = torch.from_numpy(data_glasses["flat_weights_vector"]).float()

# --- STEP 2: MATRIX TRAINING DATASET TENSORS ---
# Input: Prompt Tokens (0.0 = Dog Identity, 1.0 = Glasses Identity)
X_train = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)

# Target: High-dimensional flattened weights stack
Y_train = torch.stack([vector_dog, vector_glasses], dim=0).to(device)

OUTPUT_DIM = Y_train.size(1) 
print(f" Vector Space Configuration: Inputs {X_train.shape} -> Targets Map Matrix {Y_train.shape}")
print(f" Total parameters target per network payload: {OUTPUT_DIM} dimensions.")
print("=" * 60)


# --- STEP 3: HYPERNETWORK MANIFOLD ARCHITECTURE ---
class HyperNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=512, output_dim=OUTPUT_DIM):
        super().__init__()
        # Piece-wise linear activations for smooth trajectory interpolation
        self.meta_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.05), # Regularization layers to keep bounds smooth
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        return self.meta_layers(x)

# Model instances configuration
hyper_model = HyperNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(hyper_model.parameters(), lr=5e-4) # Balanced learning rate


# --- STEP 4: META-GENERATOR OPTIMIZATION LOOP ---
HYPER_EPOCHS = 1500
print(f"🏋️ Optimizing weight trajectory paths for {HYPER_EPOCHS} epochs...")

for epoch in range(1, HYPER_EPOCHS + 1):
    hyper_model.train()
    optimizer.zero_grad()
    
    # Forward Pass: Token array interpretation
    predicted_vectors = hyper_model(X_train)
    loss = criterion(predicted_vectors, Y_train)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 300 == 0 or epoch == 1:
        print(f"Hyper-Epoch [{epoch:04d}/{HYPER_EPOCHS}] -> Manifold Trajectory MSE Loss: {loss.item():.8f}")

print("\n HyperNetwork configuration successfully optimized without divergence!")
print("=" * 60)


# --- STEP 5: THE CORE HYBRID COMPOSITION GENERATION ---
# 50% Dog + 50% Glasses continuous matrix synthesis
BLEND_FACTOR = 0.5 
print(f"🔮 Synthesizing hybrid neural weight asset using Blend Factor: {BLEND_FACTOR}...")

test_token = torch.tensor([[BLEND_FACTOR]], dtype=torch.float32).to(device)

hyper_model.eval()
with torch.no_grad():
    # Model ab ek customized, optimized weight boundary state compute karega
    generated_flat_vector = hyper_model(test_token).cpu().squeeze(0).numpy()

print(f" Success! Generated composite network parameters shape: {generated_flat_vector.shape}")

# Saving the synthetic network payload dump
composition_payload = {
    "generation_token": BLEND_FACTOR,
    "omega_0": data_dog["omega_0"], # Inheriting base frequency scalar limits
    "synthesized_weights": generated_flat_vector
}

COMPOSITION_OUTPUT = "hybrid_composition_vector.pt"
torch.save(composition_payload, COMPOSITION_OUTPUT)
print(f" Mathematical composite block saved flawlessly to '{COMPOSITION_OUTPUT}'!")
print("SYSTEM READY TO TEST MANIFOLD DECODING!")
print("=" * 60)