import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lpips
import os
import random
# Removed Accelerate imports
from datetime import timedelta
import traceback
import glob
import sys
import shutil
import copy
import math
import gc
import pickle
import time

# Set benchmark mode if desired
torch.backends.cudnn.benchmark = True

# --- Configuration ---
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "./dataset/"             # Directory containing the .pth episode files
EPISODE_FILE_PREFIX = ""       # Prefix for episode files (e.g., "chess*.pth")
FRAME_DATA_KEY = 'target_frames'    # Key in the .pth files containing frame tensors (N, C, H, W) or (N, H, W, C)
SAVE_DIR = "./models/"              # Directory to save the final model
SAVE_FILENAME = "vae_full_finetuned_stage2_v10_api_fix_corrected.pth" # Filename for the final VAE weights

# Single GPU / Training settings
BATCH_SIZE = 8                      # Batch size for training
DECODER_LEARNING_RATE = 1e-5        # Learning rate for the decoder
DECODER_EPOCHS = 1                  # Number of epochs to train the decoder (set to 1 as requested)
IMAGE_RESOLUTION = 512              # Target image resolution
MSE_WEIGHT = 1.0                    # Weight for MSE loss
LPIPS_WEIGHT = 0.2                  # Weight for LPIPS loss (set to 0 to disable)
SCHEDULER_PATIENCE = 3              # Patience for ReduceLROnPlateau scheduler
SCHEDULER_FACTOR = 0.5              # Factor for ReduceLROnPlateau scheduler
GRADIENT_CLIP_MAX_NORM = 1.0        # Max norm for gradient clipping (set to 0 to disable)
NUM_WORKERS = 12                     # Number of DataLoader workers (adjust based on your system)
SEED = 42                           # Random seed

# --- Determine Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- VAE Decoder-Only Fine-tuning (Single GPU) ---")
print(f"Using device: {DEVICE}")

def log_memory_usage(stage_step_info=""):
    """Logs memory usage for the current CUDA device."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(DEVICE) / 1024**2
        reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(DEVICE) / 1024**2
        print(
            f"MEM {stage_step_info} | "
            f"Alloc: {allocated:.2f} MB | Max Alloc: {max_allocated:.2f} MB | "
            f"Reserv: {reserved:.2f} MB | Max Reserv: {max_reserved:.2f} MB"
        )

class ConsolidatedEpisodeDataset(Dataset):
    """
    Loads frames sequentially from multiple .pth files containing episode data.
    Modified for single-process use (removed accelerator dependencies).
    """
    def __init__(self, data_dir, episode_prefix="", image_resolution=512, frame_key='target_frames'):
        self.data_dir = data_dir
        self.episode_prefix = episode_prefix
        self.target_chw_shape = (3, image_resolution, image_resolution)
        self.target_hwc_shape = (image_resolution, image_resolution, 3)
        self.frame_key = frame_key
        print(f"Dataset using frame key: '{self.frame_key}'")
        self.index_map = []
        self.total_frames = 0

        # Build index map directly in the main process
        print("Building index map...")
        self.index_map, self.total_frames = self._build_index_map()
        print(f"Found {self.total_frames} frames in {len(self.index_map)} files.")

        if self.total_frames == 0:
            print(f"CRITICAL ERROR: Dataset has 0 frames. Check data_dir ('{self.data_dir}') and prefix ('{self.episode_prefix}'). Exiting.", file=sys.stderr)
            raise RuntimeError("Dataset contains 0 frames.")

        print("Dataset initialized successfully.")
        self.cached_file_path = None
        self.cached_episode_data = None

    def _build_index_map(self):
        """Scans the data directory and builds an index map for frame access."""
        index_map = []
        total_frames = 0
        search_pattern = os.path.join(self.data_dir, f"{self.episode_prefix}*.pth")
        episode_files = sorted(glob.glob(search_pattern))

        if not episode_files:
            print(f"W: No files found matching pattern: '{search_pattern}'", file=sys.stderr)
            return [], 0

        file_iterator = tqdm(episode_files, desc="Scanning Episodes", leave=False)
        skipped_count = 0
        corrupt_count = 0
        format_count = 0
        shape_mismatch_count = 0

        for file_path in file_iterator:
            try:
                # Basic checks before attempting to load
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    # print(f"W: Skipping empty or non-existent file: {os.path.basename(file_path)}", file=sys.stderr)
                    skipped_count += 1
                    continue

                # Load data header or minimal info if possible, but full load is often necessary
                try:
                    # Try loading only the necessary key first if files are large and I/O is slow
                    # data_header = torch.load(file_path, map_location='cpu') # Keep full load for simplicity now
                    data = torch.load(file_path, map_location='cpu')
                except (pickle.UnpicklingError, EOFError, RuntimeError, KeyError, ValueError, AttributeError) as load_err:
                    # print(f"W: Skipping corrupt/unreadable file: {os.path.basename(file_path)} ({load_err})", file=sys.stderr)
                    corrupt_count += 1
                    continue
                except Exception as general_load_err: # Catch other potential load errors
                     print(f"W: Skipping file due to general load error: {os.path.basename(file_path)} ({general_load_err})", file=sys.stderr)
                     corrupt_count += 1
                     continue


                # Validate data format
                if not isinstance(data, dict) or self.frame_key not in data or not isinstance(data[self.frame_key], torch.Tensor):
                    # print(f"W: Skipping file with invalid format or missing key '{self.frame_key}': {os.path.basename(file_path)}", file=sys.stderr)
                    format_count += 1
                    if isinstance(data, dict) and self.frame_key in data:
                         print(f"    -> Found key '{self.frame_key}' but type is {type(data[self.frame_key])}", file=sys.stderr)
                    elif isinstance(data, dict):
                         print(f"    -> Key '{self.frame_key}' not found. Available keys: {list(data.keys())}", file=sys.stderr)
                    else:
                         print(f"    -> Loaded object is not a dict, type is {type(data)}", file=sys.stderr)

                    # Explicitly delete loaded data to free memory before continuing
                    del data
                    gc.collect()
                    continue

                frames_tensor = data[self.frame_key]
                num_frames_in_file = frames_tensor.shape[0]
                needs_permute = False

                # Validate tensor dimensions and content
                if frames_tensor.ndim != 4 or num_frames_in_file == 0:
                    # print(f"W: Skipping file with invalid tensor shape (ndim={frames_tensor.ndim}, frames={num_frames_in_file}): {os.path.basename(file_path)}", file=sys.stderr)
                    shape_mismatch_count += 1
                    del data, frames_tensor # Free memory
                    gc.collect()
                    continue

                first_frame_shape = frames_tensor.shape[1:] # Shape excluding the batch dimension

                # Check if shape matches target CHW or HWC
                if first_frame_shape == self.target_chw_shape:
                    needs_permute = False
                elif first_frame_shape == self.target_hwc_shape:
                    needs_permute = True
                else:
                    # print(f"W: Skipping file with frame shape mismatch (expected {self.target_chw_shape} or {self.target_hwc_shape}, got {first_frame_shape}): {os.path.basename(file_path)}", file=sys.stderr)
                    shape_mismatch_count += 1
                    del data, frames_tensor # Free memory
                    gc.collect()
                    continue

                # Add valid file to index map
                index_map.append((total_frames, num_frames_in_file, file_path, needs_permute))
                total_frames += num_frames_in_file

                # Clear potentially large loaded data if not needed immediately
                del data, frames_tensor
                gc.collect()


            except Exception as e:
                print(f"W: Unexpected error processing '{os.path.basename(file_path)}': {e}", file=sys.stderr)
                skipped_count += 1
                # traceback.print_exc(limit=1, file=sys.stderr) # Optional: more detailed traceback
                # Ensure cleanup even if unexpected error occurs
                if 'data' in locals(): del data
                if 'frames_tensor' in locals(): del frames_tensor
                gc.collect()
                pass # Continue to the next file

        # Print summary of skipped files
        if skipped_count > 0: print(f"  [Index Map] Skipped {skipped_count} non-existent/empty/error files.")
        if corrupt_count > 0: print(f"  [Index Map] Skipped {corrupt_count} corrupt/unreadable files.", file=sys.stderr)
        if format_count > 0: print(f"  [Index Map] Skipped {format_count} files with invalid format/key.", file=sys.stderr)
        if shape_mismatch_count > 0: print(f"  [Index Map] Skipped {shape_mismatch_count} files with invalid tensor/frame shape.", file=sys.stderr)

        return index_map, total_frames

    def __len__(self):
        """Return the total number of frames across all episodes."""
        return self.total_frames

    def _find_episode(self, idx):
        """Find the file path and frame index within that file for a global index."""
        # Use binary search if index_map becomes very large, but linear scan is fine for moderate numbers
        for global_start_idx, num_frames, file_path, needs_permute in self.index_map:
            if global_start_idx <= idx < global_start_idx + num_frames:
                return file_path, idx - global_start_idx, needs_permute
        # This should ideally not happen if __len__ is correct and idx is in bounds
        raise IndexError(f"Index {idx} out of bounds or not found in index map. Total frames: {self.total_frames}")

    def __getitem__(self, idx):
        """Retrieve a single frame tensor by its global index."""
        if not (0 <= idx < self.total_frames):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self.total_frames}")

        file_path, frame_index_in_file, needs_permute = "Unknown", -1, False
        try:
            # Find the correct file and the index within that file
            file_path, frame_index_in_file, needs_permute = self._find_episode(idx)

            # Use cached data if loading the same file consecutively
            episode_data = None
            if file_path == self.cached_file_path and self.cached_episode_data is not None:
                episode_data = self.cached_episode_data
            else:
                # Load the episode data from the file
                try:
                    # Use map_location='cpu' to avoid loading directly to GPU memory if workers are used
                    episode_data = torch.load(file_path, map_location='cpu')
                except (pickle.UnpicklingError, EOFError, RuntimeError, KeyError, ValueError, AttributeError) as load_err:
                     # Indicate a problem loading this specific file if needed
                     raise TypeError(f"Corrupt file encountered during getitem: {os.path.basename(file_path)}") from load_err
                except Exception as general_load_err:
                     raise RuntimeError(f"General error loading file during getitem: {os.path.basename(file_path)}") from general_load_err


                # Basic validation after load (redundant with build_index_map but safer)
                if not isinstance(episode_data, dict) or self.frame_key not in episode_data or not isinstance(episode_data[self.frame_key], torch.Tensor):
                    # Attempt to get more info for debugging
                    loaded_type = type(episode_data)
                    keys_info = f"Keys: {list(episode_data.keys())}" if isinstance(episode_data, dict) else "Not a dict."
                    frame_key_type = type(episode_data.get(self.frame_key, None)) if isinstance(episode_data, dict) else "N/A"
                    raise TypeError(f"Invalid format in {os.path.basename(file_path)}: Loaded type {loaded_type}, {keys_info}, Frame key '{self.frame_key}' type {frame_key_type}")


                # Cache the loaded data
                self.cached_file_path = file_path
                self.cached_episode_data = episode_data

            # Extract the specific frame
            # Add checks for frame_index_in_file bounds
            if not (0 <= frame_index_in_file < len(episode_data[self.frame_key])):
                raise IndexError(f"Frame index {frame_index_in_file} out of bounds for file {os.path.basename(file_path)} with {len(episode_data[self.frame_key])} frames.")

            image_tensor = episode_data[self.frame_key][frame_index_in_file]

            # Permute if necessary (from HWC to CHW)
            if needs_permute:
                image_tensor = image_tensor.permute(2, 0, 1).contiguous() # HWC -> CHW

            # Final shape check
            if image_tensor.shape != self.target_chw_shape:
                 # This indicates an issue either in _build_index_map logic or file content inconsistency
                 raise ValueError(f"Shape mismatch for frame {idx} in file {os.path.basename(file_path)}. Got {image_tensor.shape}, expected {self.target_chw_shape}")

            # Normalize to [-1, 1] and ensure float type
            # The data should ideally be pre-normalized, but clamp for safety.
            return torch.clamp(image_tensor.float(), -1.0, 1.0)

        except Exception as e:
             # Handle errors gracefully, especially in multiprocessing contexts (DataLoader workers)
             worker_info = torch.utils.data.get_worker_info()
             worker_id = worker_info.id if worker_info else 0
             print(f"\nDataLoader Worker Error (W{worker_id}) Idx {idx}: {type(e).__name__} - {e}", file=sys.stderr)
             print(f"  File Path: {file_path}", file=sys.stderr) # file_path might be "Unknown" if _find_episode failed
             print(f"  Frame Index in File: {frame_index_in_file}", file=sys.stderr)
             # traceback.print_exc(limit=2, file=sys.stderr) # Optional detailed traceback
             # Return a dummy tensor of the correct shape and type to avoid crashing the batch collation
             return torch.zeros(self.target_chw_shape, dtype=torch.float)

if __name__ == '__main__':
    # Set random seed for reproducibility
    print(f"Setting seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Dataset and DataLoader ---
    try:
        print("Initializing Dataset...")
        dataset = ConsolidatedEpisodeDataset(DATA_DIR, EPISODE_FILE_PREFIX, IMAGE_RESOLUTION, frame_key=FRAME_DATA_KEY)
        log_memory_usage("Dataset Init")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after initialization. Please check DATA_DIR and EPISODE_FILE_PREFIX.")

        print(f"Dataset initialized. Total Frames: {len(dataset)}")
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True, # Set to True if using GPU
            prefetch_factor = 4,
            drop_last=True,  # Important for consistent batch sizes
            persistent_workers=(NUM_WORKERS > 0) # Keep workers alive between epochs
        )
        steps_per_epoch = len(dataloader)
        print(f"DataLoader initialized. Steps per epoch: {steps_per_epoch}")
        if steps_per_epoch == 0:
             print(f"WARNING: DataLoader has length 0! Effective batch size might be larger than dataset size or drop_last=True removed all batches.", file=sys.stderr)
             # Consider not exiting immediately, maybe user wants to debug dataset issues?
             # sys.exit(1) # Exit if no data can be loaded
             raise RuntimeError("DataLoader created with zero length. Cannot proceed.")


    except Exception as e:
        print(f"FATAL: Dataset/Dataloader error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- Load VAE ---
    vae = None
    vae_scale_factor = 1.0
    print(f"Loading VAE {SD_MODEL_ID}...")
    try:
        # Load VAE in float32 for stability, can potentially use bf16 if needed later and supported
        vae = AutoencoderKL.from_pretrained(SD_MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        vae_scale_factor = vae.config.scaling_factor
        print(f"VAE Scale factor: {vae_scale_factor:.4f}")
        vae = vae.to(DEVICE) # Move model to the target device
        print("VAE loaded.")
        log_memory_usage("VAE Load")
    except Exception as e:
        print(f"FATAL: VAE load failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- Load LPIPS ---
    lpips_module = None
    if LPIPS_WEIGHT > 0:
        print("Initializing LPIPS...")
        try:
            # Load LPIPS model, move to device, set to eval mode, and disable gradients
            lpips_module = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
            # Explicitly disable gradients for the LPIPS model
            for param in lpips_module.parameters():
                 param.requires_grad = False
            print("LPIPS module initialized and moved to device.")
        except Exception as e:
            print(f"E: LPIPS init failed: {e}. Disabling LPIPS loss.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            LPIPS_WEIGHT = 0 # Disable LPIPS if it fails to load
    else:
        print("LPIPS weight is 0, skipping LPIPS initialization.")

    log_memory_usage("LPIPS Init")

    # --- Configure DECODER Fine-tuning ---
    print(f"\n{'='*20} Starting DECODER Fine-tuning ({DECODER_EPOCHS} Epoch) {'='*20}")

    print("Configuring DECODER parameters...")
    num_frozen = 0
    num_unfrozen = 0
    decoder_params = []
    for name, param in vae.named_parameters():
        if name.startswith("decoder."):
            param.requires_grad_(True)
            decoder_params.append(param)
            num_unfrozen += param.numel()
        else:
            param.requires_grad_(False) # Freeze encoder and other parts like quant/post_quant conv
            num_frozen += param.numel()

    print(f"Decoder Stage: Frozen={num_frozen/1e6:.2f}M, Trainable (Decoder)={num_unfrozen/1e6:.2f}M")

    if not decoder_params:
        print(f"FATAL: No trainable decoder parameters found. Check VAE structure/naming.", file=sys.stderr)
        sys.exit(1)

    # --- Optimizer and Scheduler ---
    # Use fused AdamW if available (typically on NVIDIA GPUs with recent PyTorch/CUDA) for potential speedup
    use_fused_optimizer = (DEVICE.type == 'cuda' and hasattr(torch.optim, '_multi_tensor_adamw') and torch.cuda.is_available())
    print(f"Using Fused AdamW: {use_fused_optimizer}")
    optimizer = optim.AdamW(decoder_params, lr=DECODER_LEARNING_RATE, fused=use_fused_optimizer)

    # Corrected: Removed the 'verbose' argument from ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    print("Decoder Optimizer and Scheduler configured.")
    log_memory_usage("Decoder Optimizer Init")

    # --- Training Loop (Decoder Only) ---
    global_step = 0
    print("Starting Decoder Training Loop...")

    for epoch in range(DECODER_EPOCHS):
        epoch_num = epoch + 1
        vae.train() # Set VAE to training mode (enables dropout, etc. if any in decoder)

        # Explicitly set non-decoder parts to eval mode (redundant if requires_grad=False, but safer)
        if hasattr(vae, 'encoder'): vae.encoder.eval()
        if hasattr(vae, 'quant_conv'): vae.quant_conv.eval()
        if hasattr(vae, 'post_quant_conv'): vae.post_quant_conv.eval()


        epoch_loss_sum = {'total': 0.0, 'mse': 0.0, 'lpips': 0.0}
        num_batches_processed = 0
        last_lr_printed = optimizer.param_groups[0]['lr'] # Track LR for printing changes

        progress_bar = tqdm(enumerate(dataloader), desc=f"Decoder E{epoch_num}", total=steps_per_epoch, leave=True)

        for step, batch_images in progress_bar:

            # Basic check for valid batch from dataloader
            if not isinstance(batch_images, torch.Tensor) or batch_images.nelement() == 0 or batch_images.shape[0] != BATCH_SIZE:
                 # Handle potential issues with last batch if drop_last=False, or dataloader errors
                 if not isinstance(batch_images, torch.Tensor):
                     print(f"W: Skipping non-tensor batch at step {step+1}, type: {type(batch_images)}", file=sys.stderr)
                 elif batch_images.nelement() == 0:
                     print(f"W: Skipping empty tensor batch at step {step+1}", file=sys.stderr)
                 # This check depends on drop_last=True, remove if drop_last=False
                 # elif batch_images.shape[0] != BATCH_SIZE:
                 #    print(f"W: Skipping incomplete batch (size {batch_images.shape[0]}) at step {step+1}", file=sys.stderr)
                 continue # Skip this batch


            # Check for dummy tensors (all zeros) returned by dataset's error handling
            if (batch_images.abs().sum(dim=[1, 2, 3]) == 0).any():
                 zero_indices = (batch_images.abs().sum(dim=[1, 2, 3]) == 0).nonzero(as_tuple=True)[0]
                 print(f"W: Skipping batch containing dummy tensors (indices: {zero_indices.tolist()}) at step {step+1}", file=sys.stderr)
                 continue


            target_images_batch = batch_images.to(DEVICE).float() # Move batch to device

            try:
                # --- Forward Pass ---
                # 1. Encode image to latent space (no gradients needed for encoder)
                with torch.no_grad(): # Ensure no gradients are computed for the encoder
                    # Check if VAE output is dictionary or specific DiagonalGaussianDistribution object
                    encoder_output = vae.encode(target_images_batch) # Output is DiagonalGaussianDistribution
                    # Sample from the distribution
                    latents = encoder_output.latent_dist.sample()
                    # Apply scaling factor AFTER sampling
                    latents = latents * vae_scale_factor

                # 2. Decode latent space back to image (gradients required for decoder)
                # Ensure latents are float32 for the decoder and un-scale before passing
                decoder_output = vae.decode(latents.float() / vae_scale_factor) # Output is AutoencoderKLOutput(sample=...)
                reconstructed_images = decoder_output.sample # Get the reconstructed image samples

                # --- Loss Calculation ---
                mse_loss_val = torch.tensor(0.0, device=DEVICE)
                lpips_loss_val = torch.tensor(0.0, device=DEVICE)

                # Ensure both tensors are float32 for loss calculation
                target_f32 = target_images_batch # Already float
                recon_f32 = reconstructed_images.float()

                # MSE Loss
                if MSE_WEIGHT > 0:
                    mse_loss_val = F.mse_loss(recon_f32, target_f32, reduction="mean")

                # LPIPS Loss
                if lpips_module is not None and LPIPS_WEIGHT > 0:
                    try:
                        # LPIPS calculation should not require gradients w.r.t. its parameters
                        # Inputs need to be in range [-1, 1] for LPIPS (alexnet)
                        lpips_input1 = torch.clamp(recon_f32, -1.0, 1.0)
                        lpips_input2 = torch.clamp(target_f32, -1.0, 1.0) # Should already be clamped by dataset

                        # Ensure inputs are on the same device as the LPIPS model (already handled)
                        lpips_loss_val = lpips_module(lpips_input1, lpips_input2).mean() # Calculate mean LPIPS across batch

                        # Move the scalar loss value back to the main compute device if necessary (already handled)

                    except Exception as lpips_err:
                        print(f"W: LPIPS calculation failed at Dec E{epoch_num} S{step+1}: {lpips_err}", file=sys.stderr)
                        traceback.print_exc(limit=1, file=sys.stderr) # Optional traceback
                        lpips_loss_val = torch.tensor(0.0, device=DEVICE) # Assign zero loss if calculation fails

                # Total Loss
                total_loss = (MSE_WEIGHT * mse_loss_val) + (LPIPS_WEIGHT * lpips_loss_val)

                # --- Backward Pass and Optimization ---
                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    print(f"W: NaN/Inf loss detected at Dec E{epoch_num} S{step+1} (MSE: {mse_loss_val.item()}, LPIPS: {lpips_loss_val.item()}). Skipping step.", file=sys.stderr)
                    optimizer.zero_grad(set_to_none=True) # Zero grads even if skipping step
                    continue

                # Zero gradients before backward pass
                optimizer.zero_grad(set_to_none=True) # More efficient than setting to zero tensors

                # Compute gradients
                total_loss.backward()

                # Gradient Clipping (optional)
                if GRADIENT_CLIP_MAX_NORM > 0:
                    # Clip gradients only for the parameters being optimized (decoder_params)
                    torch.nn.utils.clip_grad_norm_(decoder_params, GRADIENT_CLIP_MAX_NORM)

                # Update weights
                optimizer.step()

                # --- Logging and Tracking ---
                # Use .item() to get Python floats and avoid holding onto graph references
                epoch_loss_sum['mse'] += mse_loss_val.item()
                epoch_loss_sum['lpips'] += lpips_loss_val.item() if LPIPS_WEIGHT > 0 else 0.0
                epoch_loss_sum['total'] += total_loss.item()
                num_batches_processed += 1
                global_step += 1

                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "MSE": f"{mse_loss_val.item():.4f}",
                    "LPIPS": f"{lpips_loss_val.item():.4f}" if LPIPS_WEIGHT > 0 else "N/A",
                    "LR": f"{current_lr:.2e}",
                    "Step": global_step
                })

            # --- Exception Handling for Training Step ---
            except Exception as e:
                print(f"\nERROR during training step Dec E{epoch_num} S{step+1}: {type(e).__name__} - {e}", file=sys.stderr)
                traceback.print_exc(limit=5, file=sys.stderr)
                try:
                    # Attempt to zero gradients even if an error occurred mid-step
                    optimizer.zero_grad(set_to_none=True)
                except Exception as zero_grad_e:
                    print(f"E: Failed to zero gradients after error: {zero_grad_e}", file=sys.stderr)

                # Optional: Add a small delay or specific handling for CUDA errors
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    print("E: CUDA Out of Memory encountered. Try reducing batch size.", file=sys.stderr)
                    # Consider exiting or trying to recover (recovery is hard)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    # sys.exit(1) # Exit on OOM?
                continue # Skip to next batch


        # --- End of Epoch ---
        progress_bar.close()

        if num_batches_processed > 0:
            avg_epoch_total = epoch_loss_sum['total'] / num_batches_processed
            avg_epoch_mse = epoch_loss_sum['mse'] / num_batches_processed
            avg_epoch_lpips = epoch_loss_sum['lpips'] / num_batches_processed if LPIPS_WEIGHT > 0 else 0.0

            print(f"--- Decoder Epoch {epoch_num} Summary ---")
            print(f"  Avg Loss: Total={avg_epoch_total:.4f}, MSE={avg_epoch_mse:.4f}" +
                  (f", LPIPS={avg_epoch_lpips:.4f}" if LPIPS_WEIGHT > 0 else "") +
                  f" (Batches: {num_batches_processed})")

            # Step the scheduler based on the average epoch loss
            current_lr_before_step = optimizer.param_groups[0]['lr']
            try:
                scheduler.step(avg_epoch_total)
                current_lr_after_step = optimizer.param_groups[0]['lr']
                if current_lr_after_step < current_lr_before_step:
                     print(f"  Learning rate reduced by scheduler to {current_lr_after_step:.2e}")

            except Exception as sched_e:
                print(f"W: Decoder scheduler step failed: {sched_e}", file=sys.stderr)

        else:
            print(f"--- Decoder Epoch {epoch_num}: W: No batches processed in this epoch ---")

        log_memory_usage(f"End Dec Epoch {epoch_num}")
        gc.collect() # Clean up memory at end of epoch
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    print(f"\n{'='*20} DECODER Fine-tuning Finished {'='*20}")

    # --- Save Final Model ---
    print(f"Saving FINAL fine-tuned VAE state dict...")
    save_path_final = os.path.join(SAVE_DIR, SAVE_FILENAME)
    try:
        # Ensure the model is in evaluation mode for consistent saving, although state_dict doesn't depend on mode
        vae.eval()
        # Save the entire state dict (including frozen encoder and fine-tuned decoder)
        final_state_dict = vae.state_dict()

        if not final_state_dict:
             print("CRITICAL ERROR: Final state dict is empty! Skipping save.", file=sys.stderr)
        else:
            torch.save(final_state_dict, save_path_final)
            print(f"Final VAE state_dict saved to: {save_path_final}")

    except Exception as save_e:
        print(f"E: Error saving FINAL checkpoint: {save_e}", file=sys.stderr)
        traceback.print_exc(limit=2, file=sys.stderr)

    print(f"\n{'='*15} Training Finished {'='*15}")
    log_memory_usage("End of Script")
    print("--- Script Finished ---")
    sys.exit(0)
