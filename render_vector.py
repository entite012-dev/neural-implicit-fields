import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 📊 1. DIRECTLY LOAD VECTORS FROM EXTRACT_CSV RUN DATA
# ==============================================================================
head_radius = 0.3868
ear_spacing = 0.2384
ear_height = 0.3184
fur_density_wave = 64.9567  # Spatial Wave Frequency (ω)

# ==============================================================================
# 🗺️ 2. GENERATE CONTINUOUS SPATIAL GRID MATRIX
# ==============================================================================
GRID_SIZE = 512
x = np.linspace(-1, 1, GRID_SIZE)
y = np.linspace(-1, 1, GRID_SIZE)
grid_x, grid_y = np.meshgrid(x, y)

# ==============================================================================
# 📐 3. MATHEMATICAL SHAPE BOUNDARIES (CONTROLS & SHAPES)
# ==============================================================================
# Head boundary circular mask
head_mask = (np.sqrt(grid_x**2 + (grid_y + 0.1)**2) <= head_radius).astype(float)

# Left and Right Ears spatial offsets
left_ear = (((grid_x + ear_spacing)**2 + (grid_y - ear_height)**2) <= (head_radius * 0.45)**2).astype(float)
right_ear = (((grid_x - ear_spacing)**2 + (grid_y - ear_height)**2) <= (head_radius * 0.45)**2).astype(float)

# Combine geometric zones into a solid skeletal mask
structural_mask = np.clip(head_mask + left_ear + right_ear, 0.0, 1.0)

# ==============================================================================
# 🌊 4. APPLY THE EXTRACTED SINE-COSINE FUR WAVE FREQUENCY
# ==============================================================================
# Building high-frequency patterns using extracted spatial frequency
sine_component = np.sin(fur_density_wave * grid_x)
cosine_component = np.cos(fur_density_wave * grid_y)
fur_interference = sine_component * cosine_component

# Masking the wave matrix inside the mathematical shape boundary
final_vector_render = structural_mask * fur_interference

# ==============================================================================
# 🎨 5. RENDER THE SPECTRUM OUT VIA MATPLOTLIB
# ==============================================================================
plt.figure(figsize=(6, 6))
plt.imshow(final_vector_render, cmap='viridis')
plt.axis('off')
plt.title(f"Raw Vector Field Geometry\nFrequency Matrix: {fur_density_wave} Hz", color='white', fontsize=12)
plt.gcf().patch.set_facecolor('black')
plt.show()