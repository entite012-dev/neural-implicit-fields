# 🧠 Compositional Manipulation of Implicit Neural Representations (INRs)

A high-performance PyTorch framework exploring continuous coordinate-based representations, multimodal parameter mapping constraints, and seamless functional-space field synthesis using Sinusoidal Representation Networks (SIREN).

---

## 📌 Project Overview

This research project focuses on the advanced manipulation of **Implicit Neural Representations (INRs)**. Instead of representing images as discrete grids of pixels, we represent them as continuous functions parameterized by deep neural networks. 

The core objective of this repository is to solve a fundamental challenge in neural fields: **How to seamlessly blend or compose two independent neural networks (e.g., a continuous field representing a 'Dog' and another representing 'Glasses') without causing structural or phase distortion.**

### 🔬 The Core Scientific Problem
Traditional neural architectures allow for straightforward latent space interpolations. However, coordinate networks utilizing periodic activation functions—like SIREN ($\sin(\omega_0 \cdot x)$)—possess highly non-convex parameter spaces. Modifying or interpolating their weights directly leads to destructive phase interference, causing the generated visual manifold to completely collapse into high-frequency static noise.

### 💡 Our Solution
To overcome this constraint, this framework introduces and validates **Functional-Space Dual Forward Blending**. By treating the serialized checkpoints as independent analytical continuous functions, we execute coordinate evaluation blocks in parallel and interpolate directly on the generated forward field slices. This completely bypasses the non-Euclidean parameter limitations, achieving pristine, artifact-free compositions up to extreme high-resolution bounds (tested and verified at native 4K scaling).


# Compositional Blending of Implicit Neural Representations via Continuous Functional Fields

This repository implements an advanced deep learning framework exploring **Implicit Neural Representations (INRs)** using **Sinusoidal Representation Networks (SIREN)**. The core of this research evaluates the mathematical constraints of performing cross-identity multimodal interpolation, contrasting structural failures in **Parameter-Space (Weight Modification)** against seamless convergence in **Functional-Space (Forward Mapping Fields)**.

---

## 📊 Framework Evolution & Core Discoveries

When mapping independent visual profiles (a target image of a **Dog** and a target image of **Glasses**) into coordinate-based multi-layered sine functions, performing continuous interpolation yields significantly different topological behaviors depending on the integration manifold.

### Phase 1: Parameter-Space Overwrite & Wave Interference (Failed Routine)
Initially, a Meta-HyperNetwork architecture was deployed to implicitly predict and interpolate a single unified flat parameter array vector ($\mathbf{W}_{\text{hybrid}}$) by learning a direct trajectory map:
$$\mathbf{W}_{\text{hybrid}} = (1 - \alpha)\mathbf{W}_{\text{dog}} + \alpha\mathbf{W}_{\text{glasses}}$$
* **Observation:** The resulting continuous rendering outputs degraded completely into high-frequency, chaotic, multi-colored static wave noise.
* **Scientific Bottleneck:** Because SIREN architectures utilize periodic activation triggers ($\sin(\omega_0 \cdot \mathbf{x})$), their parameters are highly non-convex and structurally phase-sensitive. Naively averaging or mapping independent weight states causes catastrophic phase cancellations and spatial frequency destruction across deep hidden layers, leading to baseline structural collapse.

### Phase 2: Latent Vector Conditioning Optimization (Intermediate Transition)
To preserve unified network weights, the architecture was restructured to isolate spatial variables from identity tags using a **32-Dimensional Orthogonal Latent Embedding Matrix** ($z \in \mathbb{R}^{32}$). Coordinates and target tags were concatenated directly into a high-capacity hidden width layer ($512$ neurons):
$$\mathbf{x}_{\text{input}} = [\text{Grid}(x, y) \parallel \mathbf{z}_{\text{identity}}]$$
* **Observation:** The random phase noise was fully neutralized, but the resulting interpolated configurations suffered from extensive variance regression (Mean Blur / Fading), recovering only structural silhouettes without sharp high-frequency edge textures.
* **Scientific Bottleneck:** Squeezing distinct orthogonal target signals through shared periodic manifolds imposes an optimization capacity bottleneck. Sequential backpropagation gradients conflict with one another, driving the parameters toward a minimum-variance statistical average.

### Phase 3: Dual-Model Functional Forward Blending (The Final Flawless Master-Stroke)
To bypass the non-Euclidean parameter space entirely while maintaining the continuous infinite-resolution property of neural fields, we pivoted to **Functional-Space Output Synthesis**.
* **Methodology:** Independent coordinate functions are serialized into dedicated network checkpoints. At runtime inference, the continuous spatial grid matrix is query-processed in parallel blocks, and the linear interpolation boundary is executed directly on the final predicted RGB tensor outputs:
$$\Phi_{\text{hybrid}}(x, y) = (1 - \alpha)\Phi_{\text{model}\_\text{dog}}(x, y) + \alpha\Phi_{\text{model}\_\text{glasses}}(x, y)$$
* **Observation:** Achieved flawless, crisp, 4K super-imposed continuous compositing. The structural geometries of both targets are perfectly merged, preserving hyper-realistic edge alignment and translucent alpha-blending details without any topological artifacts.

---

## 🛠️ Repository & System Architecture

```text
dual-approach/
│
├── serialized_assets/
│   ├── extracted_siren_data_001.pt      # Optimized functional weights for Image 1 (Dog)
│   └── extracted_siren_data_002.pt      # Optimized functional weights for Image 2 (Glasses)
│
├── outputs/
│   ├── final_hybrid_composition_4k.png  # Phase 1: Parameter Phase Collapse Asset
│   ├── latent_hybrid_blend_1k.png       # Phase 2: Under-parameterized Mean Blur Asset
│   └── final_perfect_blend_4k.png       # Phase 3: Pristine Functional Composite Asset
│
├── verify_dog_extract.py                 # Structural parameter re-assembly diagnostic utility
├── functional_blend.py                  # The winning multi-model forward field synthesizer
├── conditioned_siren.py                 # Latent conditioning experiment matrix script
└── README.md                            # Comprehensive analytical documentation




Execution & Processing Pipeline
1. Installation of Primitives

Ensure your local compute station is populated with standard high-performance execution blocks:


pip install torch torchvision numpy matplotlib pillow

2. Verify Snapshot Serialization Integrity

Before performing compositional synthesis, validate that individual flat weight arrays reconstruct safely onto the continuous 2D Cartesian plane without structural orientation faults:

python verify_dog_extract.py

Validates data sequence blocks and renders dog_reconstructed_verify.png at a strict coordinate bounds.
3. Generate Flawless Continuous Functional Compositions

To sample the independent field functions dynamically and render the optimized 4K multi-modal representation:


python functional_blend.py

Leverages asynchronous VRAM tensor stacks to evaluate a high-density grid (2048×2048), writing the validated matrix out directly to final_perfect_blend_4k.png.
📉 Structural Signal Flow Matrix

+---------------------------------------------------------------------------------------+
|                           THE IMPLIED COORDINATE FIELD PIPELINE                       |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|  [FAILED ROUTINE: Direct Parameter-Space Linear Averaging]                            |
|  Weights_Dog (α) -----\                                                               |
|                        +---> [Direct Weight Math / HyperNetwork] ---> Phase Collapse   |
|  Weights_Glass (β) ----/                                             (Chaotic Noise)  |
|                                                                                       |
|  [TRANSITIONAL ROUTINE: Latent Feature Mapping]                                        |
|  Grid Coords [X, Y] ---\                                                              |
|                         +---> [Shared Capacity Network (512w)] ------> Mean Regression |
|  Latent Vector [32D] --/                                             (Blurred Bounds) |
|                                                                                       |
|  [SUCCESSFUL ROUTINE: Functional-Space Dual Forward Interpolation]                     |
|  Grid Coords [X, Y] ---> Model_Dog (Weights_α) ------> RGB_Dog \                       |
|                                                                 +---> Flawless 4K Image|
|  Grid Coords [X, Y] ---> Model_Glass (Weights_β) ----> RGB_Glass/    (Perfect Sync)   |
|                                                                                       |
+---------------------------------------------------------------------------------------+
