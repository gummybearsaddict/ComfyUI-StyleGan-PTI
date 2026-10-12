import os
import copy
import pickle
import tqdm
import torch
import torch.nn.functional as F

import folder_paths
from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar
from .projector import load_vgg


@torch.inference_mode(mode=False)
@torch.enable_grad()
def tune_pivotal(
    G,
    targets: list,
    num_steps: int = 250,
    learning_rate: float = 3e-4,
    l1_weight: float = 1.0,
    lpips_weight: float = 1.0,
    save_checkpoint: bool = False,
    checkpoint_name: str = "stylegan_tuned.pkl",
    device: torch.device = torch.device('cuda:0')
):
    """
    Pivotal Tuning Inversion (PTI) for StyleGAN.
    Fine-tunes G.synthesis parameters on one or more (image, latent) pairs so that
    the generator faithfully reconstructs and blends the target images.
    """
    if len(targets) == 0:
        return G

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Pivotal Tuning] Starting fine-tuning on {len(targets)} image(s) for {num_steps} steps...")

    # Deepcopy generator and enable training gradients exclusively on G.synthesis
    G_tuned = copy.deepcopy(G).train().to(device)
    for param in G_tuned.parameters():
        param.requires_grad = False
    for param in G_tuned.synthesis.parameters():
        param.requires_grad = True

    # Setup Adam optimizer
    optimizer = torch.optim.Adam(G_tuned.synthesis.parameters(), lr=learning_rate)

    # Load perceptual loss model (VGG16)
    vgg16 = load_vgg(device)

    # Preprocess targets and compute target VGG features
    prepared_targets = []
    for item in targets:
        # Expected target image shape: [3, H, W], range [0, 255]
        tgt_img = item['image'].to(device).to(torch.float32)
        if tgt_img.ndim == 3:
            tgt_img = tgt_img.unsqueeze(0) # [1, 3, H, W]
        elif tgt_img.ndim == 4 and tgt_img.shape[0] > 1:
            tgt_img = tgt_img[-1:]

        # Compute true normalized [-1, 1] target BEFORE any processing (crucial: avoid in-place corruption)
        tgt_norm = (tgt_img.clone() / 127.5) - 1.0

        # Inverted latent code ws: shape [1, num_ws, 512]
        ws = item['latent'].to(device).to(torch.float32)
        if ws.ndim == 2:
            ws = ws.unsqueeze(0)
        # Slicing defense: if user connected training_latents (shape [N, num_ws, 512]) instead of final_latent
        if ws.shape[0] > 1:
            print(f"[Pivotal Tuning] Notice: Received latent with batch size {ws.shape[0]} (likely from 'training_latents'). Auto-slicing to final latent [-1:] to prevent GPU Out Of Memory!")
            ws = ws[-1:]
        # Ensure latent matches mapping num_ws if needed
        if ws.shape[1] == 1 and hasattr(G_tuned.mapping, 'num_ws'):
            ws = ws.repeat([1, G_tuned.mapping.num_ws, 1])

        # Compute VGG features for target image using a CLONE because TorchScript vgg16 modifies input in-place
        tgt_for_vgg = tgt_img.clone()
        if tgt_for_vgg.shape[2] > 256:
            tgt_for_vgg = F.interpolate(tgt_for_vgg, size=(256, 256), mode='area')
        with torch.no_grad():
            tgt_features = vgg16(tgt_for_vgg, resize_images=False, return_lpips=True)

        prepared_targets.append({
            'target_norm': tgt_norm,
            'target_features': tgt_features,
            'ws': ws
        })

    pbar = None
    if PROGRESS_BAR_ENABLED and num_steps > 1:
        pbar = ProgressBar(num_steps)

    tq_bar = tqdm.trange(num_steps, desc="Pivotal Tuning")

    for step in tq_bar:
        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0

        for item in prepared_targets:
            ws = item['ws']
            # Synthesize image from latent (dynamic range [-1, 1])
            synth_images = G_tuned.synthesis(ws, noise_mode='const')

            # Pixel L1 Loss directly in [-1, 1] normalized space (anchors true colors and details)
            l1_loss = F.l1_loss(synth_images, item['target_norm'])

            # Dynamic range [-1, 1] -> [0, 255] for VGG perceptual loss
            synth_images_255 = (synth_images + 1.0) * (255.0 / 2.0)
            if synth_images_255.shape[2] > 256:
                synth_images_255 = F.interpolate(synth_images_255, size=(256, 256), mode='area')

            # Clone before passing to vgg16 so in-place operation does not corrupt synth_images_255
            synth_vgg = synth_images_255.clone()
            synth_features = vgg16(synth_vgg, resize_images=False, return_lpips=True)
            lpips_loss = (item['target_features'] - synth_features).square().sum()

            # Combined weighted loss per image
            loss = (lpips_weight * lpips_loss + l1_weight * l1_loss) / len(prepared_targets)
            loss.backward()
            total_loss_val += loss.item()

        optimizer.step()

        tq_bar.set_postfix_str(f"loss: {total_loss_val:<6.3f}")
        if pbar is not None:
            pbar.update(1)

    # Freeze tuned generator and set to eval mode
    G_tuned.eval().requires_grad_(False)

    # Optional checkpoint save
    if save_checkpoint and checkpoint_name:
        if not checkpoint_name.endswith('.pkl'):
            checkpoint_name += '.pkl'
        
        # Determine destination folder
        models_dir = os.path.join(folder_paths.models_dir, "stylegan")
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, checkpoint_name)

        print(f"[Pivotal Tuning] Saving tuned StyleGAN model to: {save_path}")
        save_data = {
            'G_ema': copy.deepcopy(G_tuned).eval().cpu()
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print("[Pivotal Tuning] Save complete.")

    # Free optimizer and intermediate VRAM
    del optimizer, vgg16, prepared_targets
    torch.cuda.empty_cache()

    return G_tuned
