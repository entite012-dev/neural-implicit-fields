import torch
import numpy as np
import ollama
import json
import matplotlib.pyplot as plt

# Ensure GPU execution on RTX 3050 (6GB VRAM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Vector Engine on: {device}")

# ==========================================
# STEP 1: QWEN 3.5 9B LOGIC DEPLOYMENT
# ==========================================
def parse_dog_prompt_with_qwen(prompt_text):
    print(" Qwen 3.5 9B converting text prompt to tensor keys...")
    
    system_instruction = (
        "You are a geometry-texture generation node. Extract keys and respond ONLY with a raw JSON. "
        "Keys: 'head_x' (float -0.8 to 0.8), 'head_y' (float -0.8 to 0.8), 'ear_rotation' (float 0 to 90), "
        "'fur_frequency' (float 20 to 120), 'fur_detail_noise' (float 0.1 to 1.0). "
        "Respond with just the raw JSON string, no chat."
    )
    
    response = ollama.generate(
        model='qwen3.5:9b',
        system=system_instruction,
        prompt=prompt_text
    )
    
    clean_json = response['response'].strip()
    if clean_json.startswith("```"):
        clean_json = "".join(clean_json.split("\n")[1:-1]).strip()
        
    try:
        return json.loads(clean_json)
    except:
        print("⚠️ Qwen response mismatch, deploying baseline retriever parameters.")
        return {"head_x": 0.2, "head_y": -0.1, "ear_rotation": 35.0, "fur_frequency": 60.0, "fur_detail_noise": 0.5}

# ==============================================================================
# STEP 2 & 3: VECTOR SKELETON & FUR FREQUENCY GENERATION
# ==========================================
def generate_stylized_implicit_dog(params, grid_size=512):
    # Load LLM parsed configuration keys
    hx, hy = params['head_x'], params['head_y']
    fur_freq = params['fur_frequency']
    fur_noise = params['fur_detail_noise']
    ear_rot_deg = params['ear_rotation']

    # Initialize Coordinate Grid Tensor on GPU
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # --- Geometry Stream: Dog Skeleton via Conic Sections ---
    # f(x) = ax + b (Affine Shift to head position)
    dx = grid_x - hx
    dy = grid_y - hy
    
    # 1. Main Head Curve (Vector boundary)
    head_curve = (torch.sqrt(dx**2 + (dy/0.8)**2) <= 0.4).float()
    
    # 2. Ear Geometry & Affine Rotation Core
    theta = torch.tensor(ear_rot_deg * np.pi / 180.0, device=device)
    ex_l = torch.cos(theta) * (dx + 0.3) - torch.sin(theta) * (dy + 0.1)
    ey_l = torch.sin(theta) * (dx + 0.3) + torch.cos(theta) * (dy + 0.1)
    
    # Ear Vector Boundaries (Skeleton shapes)
    ear_l = (torch.sqrt((ex_l/0.6)**2 + ey_l**2) <= 0.2).float()
    
    # Nose & Eyes (Small Conic Nodes)
    nose = (torch.sqrt((dx/0.5)**2 + (dy+0.1)**2) <= 0.08).float()
    eyes = (torch.sqrt(((dx-0.15)/0.4)**2 + (dy-0.15)**2) <= 0.04).float() | \
           (torch.sqrt(((dx+0.15)/0.4)**2 + (dy-0.15)**2) <= 0.04).float()

    # Fuse Geometry Boundary Skeleton Mask
    dog_skeleton_mask = (head_curve + ear_l - nose - eyes)
    dog_skeleton_mask = torch.clamp(dog_skeleton_mask, 0.0, 1.0)

    # --- Texture Stream: Fur Dynamics via SIREN Fields ---
    # Complex Harmonic Waves create micro-details in phase space
    W = fur_freq / 2.0  # Base texture frequency
    W_prime = fur_freq  # Micro fur noise
    
    # Fur density field based on sine-wave flux distribution
    harmonic_fur_field = torch.sin(W * dx + fur_noise * torch.cos(W_prime * dy)) + \
                         torch.cos(W * dy + fur_noise * torch.sin(W_prime * dx))
                         
    # Phase shift based on local coordinates to orient fur
    fur_orientation = torch.atan2(dy, dx)
    local_fur_wave = harmonic_fur_field * torch.sin(fur_orientation * np.pi)

    # --- Differentiable Gating & Fusion ---
    # The ultimate implicit image function (Phase State representation)
    return dog_skeleton_mask * local_fur_wave

# ==========================================
# PIPELINE EXECUTION
# ==========================================
user_prompt = "Generate a retriever face outline, shifted slightly bottom right, detailed floppy ear rotation, and dense fur waves."
keys = parse_dog_prompt_with_qwen(user_prompt)
print(f"Generated Tensor States: {keys}")

# Process on RTX 3050 CUDA grid
dog_implicit_graph = generate_stylized_implicit_dog(keys)

# Visualize the Infinite Resolution Output
plt.figure(figsize=(8, 8))
plt.imshow(dog_implicit_graph.cpu().numpy(), cmap='turbo')
plt.axis('off')
plt.title(f"Implicit Dog Vector: {user_prompt}", color='white', fontsize=11)
plt.gcf().patch.set_facecolor('black')
plt.show()