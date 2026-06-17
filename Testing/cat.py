import torch
import numpy as np
import ollama
import matplotlib.pyplot as plt

# Connect with your CUDA cores (RTX 3050 Accelerator)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Hardware Status: Accelerating Neural Field Grid on {device}")

# ==============================================================================
# OPTIMIZED CLASSIFICATION ENGINE (FAST TOKEN DISPATCH)
# ==============================================================================
def get_animal_matrix_preset(prompt_text):
    print(f"\n🤖 Processing context via Local Ollama Engine...")
    
    presets = {
        "cat":   {"head_radius": 0.35, "ear_spacing": 0.25, "ear_height": 0.22, "fur_density": 50.0, "cmap": "inferno"},
        "mouse": {"head_radius": 0.25, "ear_spacing": 0.35, "ear_height": 0.32, "fur_density": 95.0, "cmap": "magma"},
        "bear":  {"head_radius": 0.48, "ear_spacing": 0.28, "ear_height": 0.42, "fur_density": 30.0, "cmap": "copper"}
    }
    
    try:
        response = ollama.generate(
            model='qwen3.5:9b', 
            system="Identify the target animal. Your response must contain the exact lowercase word: 'cat', 'mouse', or 'bear'.",
            prompt=prompt_text,
            options={'temperature': 0.0, 'num_predict': 15}
        )
        
        # Raw response ko clean lowercase string mein convert karo
        detected_text = response['response'].strip().lower()
        print(f"📥 Received Local Token: '{detected_text}'")
        
        # Deep substring match: poore text mein kahin bhi word ho toh nikal lo
        if "mouse" in detected_text:
            print("🎯 Match Confirmed! Deploying MOUSE Matrix Coordinates.")
            return presets["mouse"]
        elif "bear" in detected_text:
            print("🎯 Match Confirmed! Deploying BEAR Matrix Coordinates.")
            return presets["bear"]
        elif "cat" in detected_text:
            print("🎯 Match Confirmed! Deploying CAT Matrix Coordinates.")
            return presets["cat"]
                
        print("⚠️ Token mismatch with parameters. Applying safe baseline matrix.")
        
    except Exception as e:
        print(f"⚠️ Core connection freeze ({str(e)}). Applying hardware safety defaults.")
        
    return presets["cat"]

# ==============================================================================
# DYNAMIC USER INTERFACE
# ==============================================================================
print("\n" + "=" * 60)
user_prompt = input("⌨️ Enter your visual instruction prompt (e.g., mouse, bear, cat): ")
print("=" * 60)

geo_params = get_animal_matrix_preset(user_prompt)

# ==============================================================================
# PYTORCH COORDINATE GENERATION CORE
# ==============================================================================
GRID_SIZE = 512
x = torch.linspace(-1, 1, GRID_SIZE, device=device)
y = torch.linspace(-1, 1, GRID_SIZE, device=device)
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

# Extract exact values
hr = geo_params['head_radius']
es = geo_params['ear_spacing']
eh = geo_params['ear_height']
freq = geo_params['fur_density']

# Processing Conic Space Math (Skeleton Boundary)
head_mask = (torch.sqrt(grid_x**2 + (grid_y + 0.1)**2) <= hr).float()
left_ear = ((grid_x + es)**2 + (grid_y - eh)**2 <= (hr * 0.45)**2).float()
right_ear = ((grid_x - es)**2 + (grid_y - eh)**2 <= (hr * 0.45)**2).float()
skeleton_mask = torch.clamp(head_mask + left_ear + right_ear, 0.0, 1.0)

# Sinusoidal High-Frequency Texture Field
fur_texture = torch.sin(freq * grid_x) * torch.cos(freq * grid_y)
final_output = skeleton_mask * fur_texture

# Render Graph Layout
plt.figure(figsize=(7, 7))
plt.imshow(final_output.cpu().numpy(), cmap=geo_params['cmap'])
plt.axis('off')
plt.title(f"Implicit Matrix Output: {user_prompt}", color='white', fontsize=10)
plt.gcf().patch.set_facecolor('black')
print("🎨 Grid processing finish! Initializing UI window frame.")
plt.show()