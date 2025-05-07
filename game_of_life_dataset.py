import torch
import numpy as np
import os
import math
import random
from typing import List, Dict, Union, Tuple
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import imageio # For saving GIFs
import warnings

# Suppress potential UserWarnings from matplotlib about TkAgg
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

print("Libraries installed and imported.")

# Cell 2: Configuration & Helper Functions

# --- Schema Global Variables ---
GAME_NAME = "<game_of_life>"
FRAME_SIZE: Tuple[int, int, int] = (512, 512, 3)
NORMALIZATION_METHOD = "scaling_-1_1"
TORCH_DATA_TYPE = torch.float32
ACTION_STRING = "A"
FINAL_ACTION_MAX_STEPS = "<exit>"
NORMALIZATION_SCALE_FACTOR = 127.5 # For denormalization

# --- CGoL Simulation Parameters ---
GRID_SIZE: Tuple[int, int] = (64, 64)
# --- IMPORTANT: Set generation parameters here ---
N_EPISODES_TO_GENERATE = 50 # How many episodes to create
MAX_STEPS_PER_EPISODE = 100 # How many steps per episode

# --- Output Configuration (Colab specific path) ---
BASE_OUTPUT_DIR = "./dataset"
GAME_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR)

# --- Helper Functions ---

def create_standard_grey_tensor() -> torch.Tensor:
    """Creates the standard grey image tensor (128, 128, 128) normalized."""
    grey_val = 128
    normalized_grey = (grey_val / NORMALIZATION_SCALE_FACTOR) - 1.0
    grey_image = np.full(FRAME_SIZE, normalized_grey, dtype=np.float32)
    return torch.tensor(grey_image, dtype=TORCH_DATA_TYPE)

# Pre-calculate the standard grey tensor
STANDARD_GREY_IMAGE_TENSOR = create_standard_grey_tensor()

def normalize_frame(frame_np: np.ndarray) -> torch.Tensor:
    """Normalizes a numpy frame [0, 255] to [-1, 1] and converts to tensor."""
    frame_np = frame_np.astype(np.float32)
    normalized_frame = (frame_np / NORMALIZATION_SCALE_FACTOR) - 1.0
    return torch.tensor(normalized_frame, dtype=TORCH_DATA_TYPE)


def render_grid_to_frame(grid: np.ndarray) -> torch.Tensor:
    """Renders the CGoL grid (0/1) to a normalized FRAME_SIZE tensor."""
    H, W, C = FRAME_SIZE
    grid_h, grid_w = grid.shape
    scale_h = H // grid_h
    scale_w = W // grid_w
    black = np.array([0, 0, 0], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    bool_grid = grid.astype(bool)
    colored_grid = np.zeros((grid_h, grid_w, C), dtype=np.uint8)
    colored_grid[~bool_grid] = black
    colored_grid[bool_grid] = white
    upscaled_frame_np = colored_grid.repeat(scale_h, axis=0).repeat(scale_w, axis=1)
    upscaled_frame_np = upscaled_frame_np[:H, :W, :] # Crop if needed
    return normalize_frame(upscaled_frame_np)

def initialize_grid(size: Tuple[int, int]) -> np.ndarray:
    """Initializes a random CGoL grid."""
    return np.random.choice([0, 1], size=size, p=[0.75, 0.25])

def step_grid(grid: np.ndarray) -> np.ndarray:
    """Performs one step of Conway's Game of Life using convolution."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_counts = convolve2d(grid, kernel, mode='same', boundary='wrap')
    birth = (grid == 0) & (neighbor_counts == 3)
    survival = (grid == 1) & ((neighbor_counts == 2) | (neighbor_counts == 3))
    new_grid = np.zeros_like(grid)
    new_grid[birth | survival] = 1
    return new_grid

print("Configuration and helper functions defined.")
print(f"Target output directory: {GAME_OUTPUT_DIR}")
print(f"Number of episodes to generate: {N_EPISODES_TO_GENERATE}")
print(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")

# Cell 3: Data Generation Logic

def generate_cgol_episode(episode_number: int) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Generates a single episode as consolidated tensors for CGoL."""
    # Initialize lists to collect all frames and actions
    previous_frames = []
    target_frames = []
    actions = []
    
    # 1. Initial State
    current_grid = initialize_grid(GRID_SIZE)
    F0 = render_grid_to_frame(current_grid)
    previous_frames.append(STANDARD_GREY_IMAGE_TENSOR.clone())
    actions.append(GAME_NAME)
    target_frames.append(F0)
    
    previous_frame_tensor = F0

    # 2. Simulation Steps
    last_meaningful_frame = F0
    terminated = False
    final_action = FINAL_ACTION_MAX_STEPS # Default assumption
    FN = F0 # Placeholder

    for step in range(MAX_STEPS_PER_EPISODE):
        Fi = previous_frame_tensor
        last_meaningful_frame = Fi # Store potential last meaningful frame
        next_grid = step_grid(current_grid)
        Fi_plus_1 = render_grid_to_frame(next_grid)

        previous_frames.append(Fi)
        actions.append(ACTION_STRING)
        target_frames.append(Fi_plus_1)

        current_grid = next_grid
        previous_frame_tensor = Fi_plus_1

        if step == MAX_STEPS_PER_EPISODE - 1:
            final_action = FINAL_ACTION_MAX_STEPS
            terminated = True
            FN = Fi # The frame *before* this last step led to termination
            break

    if not terminated: # Should not happen with max steps, but good practice
        print(f"Warning: Episode {episode_number} loop finished unexpectedly.")
        final_action = FINAL_ACTION_MAX_STEPS
        FN = last_meaningful_frame

    # 3. Final Step
    previous_frames.append(FN)
    actions.append(final_action)
    target_frames.append(STANDARD_GREY_IMAGE_TENSOR.clone())

    # Convert lists to stacked tensors
    episode_data = {
        'previous_frames': torch.stack(previous_frames),
        'actions': actions,  # Keeping as list since mixed types
        'target_frames': torch.stack(target_frames)
    }
    
    return episode_data

print("Episode generation function defined.")


# Cell 4: Run Data Generation

print("Starting dataset generation...")

# Ensure output directory exists
os.makedirs(GAME_OUTPUT_DIR, exist_ok=True)
print(f"Ensured output directory exists: {GAME_OUTPUT_DIR}")

generated_files = []
# Generate and save episodes
for i in range(1, N_EPISODES_TO_GENERATE + 1):
    episode_data = generate_cgol_episode(i)

    # Define filename
    filename = f"{GAME_NAME}_epi{i}.pth"
    filepath = os.path.join(GAME_OUTPUT_DIR, filename)
    generated_files.append(filepath)

    # Save the episode data
    try:
        torch.save(episode_data, filepath)
        if i % 10 == 0 or i == N_EPISODES_TO_GENERATE: # Print progress less often
             print(f"Saved: {filepath} ({len(episode_data['actions'])} transitions)")
    except Exception as e:
        print(f"Error saving episode {i}: {e}")

print("-" * 20)
print("Dataset generation complete.")
print(f"Generated {len(generated_files)} files in {GAME_OUTPUT_DIR}:")
# List first few files if many were generated
for f in generated_files[:5]:
    print(f" - {os.path.basename(f)}")
if len(generated_files) > 5:
    print("   ...")
