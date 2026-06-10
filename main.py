import torch
import numpy as np
import matplotlib.pyplot as plt
import ollama
import json

# Ensure CUDA usage for RTX 3050
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# STEP 1: Local LLM Parameter Extraction
# ---------------------------------------------------------
def get_parameters_from_prompt(prompt_text):
    system_instruction = (
        "You are a strict coordinate generator. Respond ONLY with a raw JSON object. "
        "Fields: 'center_x' (float -1 to 1), 'center_y' (float -1 to 1), "
        "'angle' (float 0 to 360), 'frequency' (float 10 to 100). No markdown, no prose."
    )
    
    response = ollama.generate(
        model='qwen3.5:9b',
        system=system_instruction,
        prompt=prompt_text
    )
    
    # JSON Parse karna
    try:
        params = json.loads(response['response'].strip())
        return params
    except:
        # Fallback agar LLM ne galat format diya
        return {"center_x": 0.0, "center_y": 0.0, "angle": 0.0, "frequency": 30.0}

# ---------------------------------------------------------
# STEP 2 & 3: Affine Transformation & Implicit Field Generation
# ---------------------------------------------------------
def generate_implicit_vector(params, grid_size=1024):
    cx, cy = params['center_x'], params['center_y']
    angle = params['angle']
    freq = params['frequency']
    
    # Coordinate grid banana on GPU
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Affine Step A: Translation (Shift: x - x0)
    shifted_x = grid_x - cx
    shifted_y = grid_y - cy
    
    # Affine Step B: Rotation Matrix Injection (f(x) = ax + b)
    theta = torch.tensor(angle * 3.14159 / 180.0, device=device)
    local_x = torch.cos(theta) * shifted_x - torch.sin(theta) * shifted_y
    local_y = torch.sin(theta) * shifted_x + torch.cos(theta) * shifted_y
    
    # Differentiable Gating: Shape constraint (Boundary Node Box)
    # Bounding condition for a localized square field
    mask = (torch.abs(local_x) <= 0.4) & (torch.abs(local_y) <= 0.4)
    mask = mask.float()
    
    # Physical Harmonics: Sinusoidal Texture Field
    texture = torch.sin(freq * local_x) * torch.cos(freq * local_y)
    
    # Ultimate Fusion
    return mask * texture

# ---------------------------------------------------------
# PIPELINE EXECUTION
# ---------------------------------------------------------
user_prompt = "A heavy frequency futuristic tech matrix centered top right, rotated steeply"
print("Processing Prompt via Local Llama-3.2...")

extracted_json = get_parameters_from_prompt(user_prompt)
print(f"Extracted Mathematical Parameters: {extracted_json}")

print("Computing Affine Projections and Tensor Fields on RTX 3050...")
vector_field = generate_implicit_vector(extracted_json)

# Plotting the Masterpiece
plt.imshow(vector_field.cpu().numpy(), cmap='terminal' if 'terminal' in plt.colormaps() else 'viridis')
plt.axis('off')
plt.title(f"MEXT Research Engine: {user_prompt}", fontsize=10, color='white')
plt.gcf().patch.set_facecolor('black')
plt.show()