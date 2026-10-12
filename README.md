# ComfyUI-StyleGan

Basic support for StyleGAN2 and StyleGAN3 models.  
![workflow](https://raw.githubusercontent.com/spacepxl/ComfyUI-StyleGan/main/workflow.png)

Original:  
https://github.com/NVlabs/stylegan3

Models:  
- **Tuned Checkpoint (`stylegan_tuned.pkl`)**: [Download from Releases](https://github.com/gummybearsaddict/ComfyUI-StyleGan-PTI/releases/download/v1.0.0/stylegan_tuned.pkl) (or visit the [v1.0.0 Release page](https://github.com/gummybearsaddict/ComfyUI-StyleGan-PTI/releases/tag/v1.0.0))
- **Official Nvidia StyleGAN2 Checkpoints**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files  
- **Official Nvidia StyleGAN3 Checkpoints**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files  

Place any models you want to use in `ComfyUI/models/stylegan/*.pkl` (create the folder if it doesn't exist).

## Installation

StyleGAN uses custom CUDA extensions which are compiled at runtime, so unfortunately the setup process can be a bit of a pain.

You need CUDA Toolkit, ninja, and either GCC (Linux) or Visual Studio (Windows). Tested on Windows with CUDA Toolkit 11.7 and VS2019 Community. You may also need to add paths to the system PATH, CUDA_HOME, and LD_LIBRARY_PATH.

```
PATH:
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build

CUDA_HOME:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7

LD_LIBRARY_PATH:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\lib\x64
```

If you're using ComfyUI portable, the embedded python installation is probably also missing some necessary files. The only solution I found to this was to just copy them from a full system installation of python 3.10.x to the embedded installation.

From `C:/Users/username/AppData/Local/Programs/Python/Python310/include/*`  
to `ComfyUI_windows_portable/python_embeded/Include/*`  
(make sure you don't overwrite any file/folders that are already there)

And from `C:/Users/username/AppData/Local/Programs/Python/Python310/libs/*`  
to `ComfyUI_windows_portable/python_embeded/libs/*`

If all of that is set up correctly, when you run a StyleGAN workflow, it will first build the necessary PyTorch plugins (should take 30-60s), then generate an image. There will be a message in the console, and then subsequent images will be much faster to generate (measured at 64 images/sec on a 3090 with a large batch, although ComfyUI's tensor to PIL for previews will bottleneck realtime generation to more like 8 fps)

StyleGAN2:  
```
Setting up PyTorch plugin "bias_act_plugin"... Done.
Setting up PyTorch plugin "upfirdn2d_plugin"... Done.
```
StyleGAN3:  
```
Setting up PyTorch plugin "bias_act_plugin"... Done.
Setting up PyTorch plugin "filtered_lrelu_plugin"... Done.
```  

## Extended Features & New Nodes

### 1. StyleGAN Pivotal Tuning (PTI)
Fine-tunes the generator weights around inverted pivot latents using a combined Perceptual LPIPS (VGG16) and Pixel L1 loss. This allows high-fidelity reconstruction, editing, and seamless latent blending between distinct images without color collapse.
- Supports up to 3 target pairs (`image_1` + `latent_1`, `image_2` + `latent_2`, `image_3` + `latent_3`).
- Configurable steps, learning rate, and loss weights (`l1_weight`, `lpips_weight`).
- Optional checkpoint exporter to save fine-tuned `.pkl` generators directly to `ComfyUI/models/stylegan/`.

### 2. Multi-Scale Blend & Interpolation (`BlendStyleGANLatents`)
Interpolate between any two arbitrary images with four distinct blending modes:
- **`smooth_layers` (Recommended default)**: Perceptual wave across generator layers (fine layers lead transition, coarse layers follow smoothly). Guarantees that **any two images blend cleanly across the entire 0.0 to 1.0 spectrum without turning into an abstract blur**.
- **`custom_zones`**: Independent control over 3 anatomical scale bands:
  - `coarse_blend` (Layers 0..4): Geometry, perspective, macro composition.
  - `medium_blend` (Layers 5..9): Objects, silhouettes, major features.
  - `fine_blend` (Layers 10..15): Lighting, color palette, micro-textures.
- **`slerp`**: Spherical linear interpolation across all latent dimensions.
- **`lerp`**: Standard linear interpolation.

### 3. Layer-Wise Style Mixing (`StyleGANStyleMixing`)
Enables surgical transplantation and independent percentage blending between two images:
- `latent_1` & `latent_2`: Any two StyleGAN latents.
- `crossover_layer`: Layer cutoff dividing structure from style (1 to 16, default: 7).
- `structure_blend`: Percentage blend of `latent_2` into the structure layers (0.0 = 100% latent_1, 1.0 = 100% latent_2).
- `style_blend`: Percentage blend of `latent_2` into the style layers (0.0 = 100% latent_1, 1.0 = 100% latent_2).

### 4. W+ and W Space Inversion with Luminance Anchor
- Select between `"W+"` (default, optimizes 16 independent layer vectors for photographic fidelity) and `"W"` (single shared vector).
- Added `pixel_weight` to `StyleGANInversion` to anchor dynamic range and color palette against dark scenes, preventing low-key collapse.
- Default steps optimized to 250 for faster convergence without latent over-fitting.

### 5. Adaptive Dynamic Range Recovery
- `StyleGANSampler` includes adaptive contrast recovery to guard against crushed shadow values ($\le -1.0$) or saturation blowouts.

## Example Workflow
An end-to-end example workflow is included in:
`workflows/Comfy_StyleGAN3_PTI.json`

Drag and drop this file into ComfyUI to load the full pipeline:
`Load Images -> Inversion (W+) -> Pivotal Tuning -> Latent Blending / Style Mixing -> Sampler -> Upscale -> Preview`
