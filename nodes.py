import os
import sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import trange

from .slerp import slerp

from . import dnnlib
from . import torch_utils
# from . import legacy
sys.modules["dnnlib"] = dnnlib
sys.modules["torch_utils"] = torch_utils
# sys.modules["legacy"] = legacy

import folder_paths
from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar

# set the models directory
if "stylegan" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "stylegan")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["stylegan"]
folder_paths.folder_names_and_paths["stylegan"] = (current_paths, folder_paths.supported_pt_extensions)

class LoadStyleGAN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_file": (folder_paths.get_filename_list("stylegan"), ),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN",)
    FUNCTION = "load_stylegan"
    CATEGORY = "StyleGAN"
    
    def load_stylegan(self, stylegan_file):
        with open(folder_paths.get_full_path("stylegan", stylegan_file), 'rb') as f:
            G = pickle.load(f)['G_ema'].cuda()
            # device = torch.device('cuda:0')
            # G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device)
        return (G,)

class GenerateStyleGANLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "class_label": ("INT", {"default": -1, "min": -1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024}),
                
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN"
    
    def generate_latent(self, stylegan_model, seed, class_label, batch_size):
        torch.manual_seed(seed)
        if class_label < 0:
            class_label = None
        
        z = torch.randn([batch_size, stylegan_model.z_dim]).cuda()
        w = []
        for i in range(batch_size):
            w.append(stylegan_model.mapping(z[i].unsqueeze(0), class_label))
        
        return (torch.cat(w, dim=0), )

class StyleGANSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "noise_mode": (['const', 'random'],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "StyleGAN"
    
    def generate_image(self, stylegan_model, stylegan_latent, noise_mode, seed):
        torch.manual_seed(seed)
        imgs = []
        batch_size = stylegan_latent.size(0)
        pbar = None
        if PROGRESS_BAR_ENABLED and batch_size > 1:
            pbar = ProgressBar(batch_size)
        for i in trange(batch_size):
            img = stylegan_model.synthesis(stylegan_latent[i].unsqueeze(0), noise_mode=noise_mode)
            img = torch.permute(img, (0, 2, 3, 1)) # BCHW -> BHWC
            img_norm = img / 2.0 + 0.5
            
            # Check for extreme saturation or dynamic range collapse (prevent crushed shadows or blown whites)
            range_span = (img_norm.max() - img_norm.min()).item()
            mean_val = img_norm.mean().item()
            if img_norm.min() >= 0.95 or img_norm.max() <= 0.05 or (range_span < 0.20 and mean_val < 0.05):
                print(f"[StyleGANSampler] Warning: Output dynamic range collapsed (min={img.min():.2f}, max={img.max():.2f}, mean={mean_val:.3f}, span={range_span:.3f}). Applying adaptive dynamic range recovery.")
                val_min = img.min()
                val_max = img.max()
                if (val_max - val_min) > 1e-5:
                    img_norm = (img - val_min) / (val_max - val_min)
                else:
                    img_norm = torch.clamp(img_norm, 0.0, 1.0)
            else:
                img_norm = torch.clamp(img_norm, 0.0, 1.0)
            imgs.append(img_norm)
            if pbar is not None:
                pbar.update(1)
        
        imgs = torch.cat(imgs, dim=0)
        return (imgs, )

class StyleGANInversion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "image": ("IMAGE", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "num_steps": ("INT", {"default": 250, "min": 1}),
                "w_avg_samples": ("INT", {"default": 10000, "min": 1, "max": 100000}),
                "initial_learning_rate": ("FLOAT", {"default": 0.1, "min": 0.00001, "max": 1.0, "step": 0.00001}),
                "initial_noise_factor": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "lr_rampdown_length": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lr_rampup_length": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_ramp_length": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "regularize_noise_weight": ("FLOAT", {"default": 30000.0, "min": 0.0, "max": 1e7}),
                "pixel_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "latent_space": (["W+", "W"], {"default": "W+"}),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT", "STYLEGAN_LATENT")
    RETURN_NAMES = ("training_latents", "final_latent")
    FUNCTION = "train_inversion"
    CATEGORY = "StyleGAN"
    
    def train_inversion(
        self,
        stylegan_model,
        image,
        seed,
        num_steps,
        w_avg_samples,
        initial_learning_rate,
        initial_noise_factor,
        lr_rampdown_length,
        lr_rampup_length,
        noise_ramp_length,
        regularize_noise_weight,
        pixel_weight=1.0,
        latent_space="W+",
        ):
        
        device = torch.device('cuda:0')
        img_resolution = stylegan_model.img_resolution
        target_image = torch.permute(image[...,:3], (0, 3, 1, 2)) # BHWC -> BCHW, RGB only
        if target_image.shape != (stylegan_model.img_channels, img_resolution, img_resolution):
            target_image = F.interpolate(target_image, size=(img_resolution, img_resolution), mode='area')
        target_image = (target_image[0] * 255.0).clone()
        
        from .projector import project
        
        seed = seed % (2**32)
        projected_w_steps = project(
            stylegan_model,
            target_image,
            num_steps = num_steps,
            w_avg_samples = w_avg_samples,
            seed = seed,
            initial_learning_rate       = initial_learning_rate,
            initial_noise_factor        = initial_noise_factor,
            lr_rampdown_length          = lr_rampdown_length,
            lr_rampup_length            = lr_rampup_length,
            noise_ramp_length           = noise_ramp_length,
            regularize_noise_weight     = regularize_noise_weight,
            pixel_weight                = pixel_weight,
            latent_space                = latent_space,
            device                      = device,
            )
        
        del stylegan_model, target_image
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return (projected_w_steps, projected_w_steps[-1].unsqueeze(0))

class BlendStyleGANLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_1": ("STYLEGAN_LATENT", ),
                "latent_2": ("STYLEGAN_LATENT", ),
                "blend": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.001}),
                "mode": (["lerp", "slerp"],),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, latent_1, latent_2, blend, mode):
        # Defensive check: if full training trajectories were connected instead of final_latent
        if latent_1.ndim == 3 and latent_1.shape[0] > 1:
            print(f"[BlendStyleGANLatents] Note: latent_1 has batch size {latent_1.shape[0]} (likely from 'training_latents'). Auto-slicing to final latent [-1:].")
            latent_1 = latent_1[-1:]
        if latent_2.ndim == 3 and latent_2.shape[0] > 1:
            print(f"[BlendStyleGANLatents] Note: latent_2 has batch size {latent_2.shape[0]} (likely from 'training_latents'). Auto-slicing to final latent [-1:].")
            latent_2 = latent_2[-1:]

        if latent_1.shape != latent_2.shape:
            raise Exception(f"latent_1 shape {latent_1.shape} and latent_2 shape {latent_2.shape} do not match!")
        
        if mode == "slerp":
            z = slerp(latent_1, latent_2, blend)
        else:
            z = torch.lerp(latent_1, latent_2, blend)
        
        return (z, )

class StyleGANStyleMixing:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_structure": ("STYLEGAN_LATENT", ),
                "latent_style": ("STYLEGAN_LATENT", ),
                "crossover_layer": ("INT", {"default": 7, "min": 1, "max": 17, "step": 1}),
            },
        }

    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "mix_styles"
    CATEGORY = "StyleGAN/extra"

    def mix_styles(self, latent_structure, latent_style, crossover_layer):
        # Defensive check: if full training trajectories were connected instead of final_latent
        if latent_structure.ndim == 3 and latent_structure.shape[0] > 1:
            latent_structure = latent_structure[-1:]
        if latent_style.ndim == 3 and latent_style.shape[0] > 1:
            latent_style = latent_style[-1:]

        if latent_structure.shape != latent_style.shape:
            raise Exception(f"latent_structure shape {latent_structure.shape} and latent_style shape {latent_style.shape} do not match!")

        total_layers = latent_structure.shape[1]
        crossover = min(max(1, crossover_layer), total_layers - 1)

        # Style Mixing: Layers 0..crossover from structure, crossover..end from style
        mixed_latent = torch.cat([
            latent_structure[:, :crossover, :],
            latent_style[:, crossover:, :]
        ], dim=1)

        print(f"[StyleGANStyleMixing] Transplanted {crossover} structure layers (0..{crossover-1}) with {total_layers - crossover} style layers ({crossover}..{total_layers-1})")
        return (mixed_latent, )

class BatchAverageStyleGANLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, stylegan_latent):
        w = torch.mean(stylegan_latent, dim=0, keepdim=True)
        std, mean = torch.std_mean(w)
        w = (w - mean) / std
        
        return (w, )

class StyleGANLatentFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_latent": ("STYLEGAN_LATENT", ),
                "index": ("INT", {"default": 0, "min": 0}),
            },
        }
    
    RETURN_TYPES = ("STYLEGAN_LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "StyleGAN/extra"
    
    def generate_latent(self, stylegan_latent, index):
        clipped_index = min(index, stylegan_latent.size(0) - 1)
        w = stylegan_latent[clipped_index].unsqueeze(0).detach().clone()
        
        return (w, )

class StyleGANPivotalTuning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stylegan_model": ("STYLEGAN", ),
                "image_1": ("IMAGE", ),
                "latent_1": ("STYLEGAN_LATENT", ),
                "steps": ("INT", {"default": 350, "min": 10, "max": 2000, "step": 10}),
                "learning_rate": ("FLOAT", {"default": 0.00035, "min": 0.00001, "max": 0.01, "step": 0.00005}),
                "l1_weight": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "lpips_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "save_checkpoint": ("BOOLEAN", {"default": True}),
                "checkpoint_name": ("STRING", {"default": "stylegan_tuned.pkl"}),
            },
            "optional": {
                "image_2": ("IMAGE", ),
                "latent_2": ("STYLEGAN_LATENT", ),
                "image_3": ("IMAGE", ),
                "latent_3": ("STYLEGAN_LATENT", ),
            }
        }
    
    RETURN_TYPES = ("STYLEGAN", )
    FUNCTION = "tune"
    CATEGORY = "StyleGAN"

    @torch.inference_mode(mode=False)
    @torch.enable_grad()
    def tune(
        self,
        stylegan_model,
        image_1,
        latent_1,
        steps,
        learning_rate,
        l1_weight,
        lpips_weight,
        save_checkpoint,
        checkpoint_name,
        image_2=None,
        latent_2=None,
        image_3=None,
        latent_3=None,
    ):
        device = torch.device('cuda:0')
        img_resolution = stylegan_model.img_resolution

        def prepare_image(img):
            tgt = torch.permute(img[..., :3], (0, 3, 1, 2)) # BHWC -> BCHW
            if tgt.shape[-2:] != (img_resolution, img_resolution):
                tgt = F.interpolate(tgt, size=(img_resolution, img_resolution), mode='area')
            return (tgt[0] * 255.0).clone()

        targets = []
        if image_1 is not None and latent_1 is not None:
            targets.append({
                'image': prepare_image(image_1),
                'latent': latent_1
            })
        if image_2 is not None and latent_2 is not None:
            targets.append({
                'image': prepare_image(image_2),
                'latent': latent_2
            })
        if image_3 is not None and latent_3 is not None:
            targets.append({
                'image': prepare_image(image_3),
                'latent': latent_3
            })

        from .pti_trainer import tune_pivotal

        tuned_G = tune_pivotal(
            G=stylegan_model,
            targets=targets,
            num_steps=steps,
            learning_rate=learning_rate,
            l1_weight=l1_weight,
            lpips_weight=lpips_weight,
            save_checkpoint=save_checkpoint,
            checkpoint_name=checkpoint_name,
            device=device,
        )

        return (tuned_G, )

NODE_CLASS_MAPPINGS = {
    "LoadStyleGAN": LoadStyleGAN,
    "GenerateStyleGANLatent": GenerateStyleGANLatent,
    "StyleGANSampler": StyleGANSampler,
    "BlendStyleGANLatents": BlendStyleGANLatents,
    "StyleGANStyleMixing": StyleGANStyleMixing,
    "BatchAverageStyleGANLatents": BatchAverageStyleGANLatents,
    "StyleGANLatentFromBatch": StyleGANLatentFromBatch,
    "StyleGANInversion": StyleGANInversion,
    "StyleGANPivotalTuning": StyleGANPivotalTuning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStyleGAN": "Load StyleGAN Model",
    "GenerateStyleGANLatent": "Generate StyleGAN Latent",
    "StyleGANSampler": "StyleGAN Sampler",
    "BlendStyleGANLatents": "Blend StyleGAN Latents (lerp or slerp)",
    "StyleGANStyleMixing": "StyleGAN Style Mixing (Layer Slicing)",
    "BatchAverageStyleGANLatents": "Batch Average StyleGAN Latents",
    "StyleGANLatentFromBatch": "StyleGAN Latent From Batch",
    "StyleGANInversion": "StyleGAN Inversion",
    "StyleGANPivotalTuning": "StyleGAN Pivotal Tuning (PTI)",
}