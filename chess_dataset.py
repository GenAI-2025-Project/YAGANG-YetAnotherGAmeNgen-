import os
import random
import torch
import chess
import chess.svg
# No engine or HF transformers needed
import cairosvg # Make sure this is installed and its dependencies are met
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt # Import Matplotlib for visualization
# Removed itertools as it's no longer needed for the old generator

# --- [1] Configuration & Constants ---

# Chess Specific & Output Schema
GAME_NAME = "<chess>"
OUTPUT_DIR = "./dataset"
DATASET_SUBDIR = "" # << NEW NAME reflecting method
N_EPISODES_DATASET = 200
FINAL_ACTION_TOKEN = "<exit>"

# --- Illegal Move Configuration ---
# Probability to *attempt* adding an illegal move derived from pseudo-legal moves
ILLEGAL_MOVE_PROBABILITY = 0.1 # Approx 1 in 10 moves will be an attempt

# Frame/Tensor Schema related
FRAME_SIZE = (512, 512, 3)  # (Height, Width, Channels)
IMAGE_SIZE = 512
TORCH_DATA_TYPE = torch.float32

# Generate the standard grey image tensor (G)
STANDARD_GREY_IMAGE_TENSOR = torch.tensor(np.full(FRAME_SIZE, 128, dtype=np.uint8), dtype=torch.float32) / 127.5 - 1.0
G = STANDARD_GREY_IMAGE_TENSOR

# --- [2] Helper Functions (Image Processing & Illegal Move Generation) ---

def pil_to_normalized_tensor(pil_image):
    """Converts PIL Image (RGB, 0-255) to Tensor (H, W, C), scaled [-1, 1]."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image = pil_image.resize((FRAME_SIZE[1], FRAME_SIZE[0]), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(pil_image)
    tensor = tensor * 2.0 - 1.0
    tensor = tensor.permute(1, 2, 0)
    return tensor.to(TORCH_DATA_TYPE)

def svg_to_pil(svg_code, output_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Converts SVG string to PIL Image (RGB)."""
    try:
        svg_bytes = svg_code.encode('utf-8')
        png_data = cairosvg.svg2png(bytestring=svg_bytes, output_width=FRAME_SIZE[1], output_height=FRAME_SIZE[0])
        image = Image.open(BytesIO(png_data)).convert("RGB")
        image = image.resize((FRAME_SIZE[1], FRAME_SIZE[0]), Image.Resampling.LANCZOS)
        return image
    except NameError: # Handle cairosvg import errors
        print("ERROR: cairosvg module not found or failed to import.")
        # Provide more specific installation instructions if possible
        print("Ensure cairosvg and its dependencies (like Cairo, Pango) are installed.")
        print("On Debian/Ubuntu: sudo apt-get install libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev pkg-config")
        print("Then install with pip: pip install cairosvg")
        raise # Re-raise the error to stop execution if critical
    except ImportError:
        print("ERROR: cairosvg installed but failed to import internal components.")
        print("This usually means system dependencies (like Cairo, Pango) are missing.")
        print("On Debian/Ubuntu: sudo apt-get install libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev pkg-config")
        raise # Re-raise the error
    except Exception as e:
        print(f"Error converting SVG to PIL: {e}")
        print("Returning grey image due to SVG conversion error.")
        grey_pil = Image.new("RGB", (FRAME_SIZE[1], FRAME_SIZE[0]), (128, 128, 128))
        return grey_pil


def tensor_to_pil(tensor):
    """Converts a (H, W, C) tensor scaled [-1, 1] back to a PIL Image (RGB)."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.to(torch.float32)
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    pil_image = transforms.ToPILImage()(tensor)
    return pil_image

# --- NEW: User-provided Illegal Move Generator ---
def generate_illegal_moves(board: chess.Board):
    """
    Finds moves that are pseudo-legal (valid piece movement) but not
    fully legal (usually because they leave the king in check).
    Returns a list of chess.Move objects.
    """
    # 1. All "pseudo-legal" moves (ignores check-status after move)
    all_pseudo = list(board.pseudo_legal_moves)
    # 2. All fully legal moves (must leave king safe)
    all_legal = list(board.legal_moves)
    # 3. Anything pseudo-legal but not legal is illegal (typically exposes king to check)
    illegal_moves = [m for m in all_pseudo if m not in all_legal]
    return illegal_moves

# --- [3] Generate Dataset (Modified for Pseudo-Legal Illegal Moves) ---

def generate_chess_episodes_random(num_episodes=N_EPISODES_DATASET, save_dir=DATASET_SUBDIR):
    """
    Generates chess episodes using random legal moves, randomly injecting
    illegal move attempts derived from pseudo-legal moves.
    Saves them in the specified format.
    """
    current_working_dir = os.getcwd()
    output_dir_abs = os.path.join(current_working_dir, OUTPUT_DIR)
    save_dir_abs = os.path.join(output_dir_abs, save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)
    print(f"Generating {num_episodes} chess episodes (with pseudo-legal illegal moves) in {save_dir_abs}...")
    total_transitions = 0
    illegal_moves_added = 0

    for epi_num in tqdm(range(1, num_episodes + 1), desc="Generating Episodes"):
        board = chess.Board() # Fresh board

        try:
            initial_svg = chess.svg.board(board=board, size=IMAGE_SIZE)
            initial_pil = svg_to_pil(initial_svg)
            if initial_pil is None:
                 print(f"ERROR: Failed to render initial board for episode {epi_num}. Skipping.")
                 continue
            F0 = pil_to_normalized_tensor(initial_pil)
        except Exception as render_err:
            print(f"FATAL: Error rendering initial board for episode {epi_num}: {render_err}. Skipping.")
            continue

        previous_frames = []
        actions = []
        target_frames = []

        previous_frames.append(G)
        actions.append(GAME_NAME)
        target_frames.append(F0)

        current_frame_tensor = F0
        episode_transitions = 1
        game_result = "*"

        while not board.is_game_over(claim_draw=True):
            # Get legal moves first, needed for both paths
            legal_moves = list(board.legal_moves)

            # --- <<< ILLEGAL MOVE INJECTION LOGIC (Using new function) >>> ---
            attempt_illegal = random.random() < ILLEGAL_MOVE_PROBABILITY
            illegal_move_generated_this_turn = False # Flag

            if attempt_illegal:
                # --- Use the specific pseudo-legal based generator ---
                pseudo_illegal_moves_list = generate_illegal_moves(board)

                if pseudo_illegal_moves_list: # Check if any were found
                    # Choose one of these specific illegal moves
                    chosen_illegal_move = random.choice(pseudo_illegal_moves_list)
                    illegal_move_uci = chosen_illegal_move.uci()
                    # print(f"Injecting illegal (pseudo-legal) move: {illegal_move_uci}") # Optional debug

                    # Store transition: Prev -> Illegal Action -> Same Frame
                    previous_frames.append(current_frame_tensor)
                    actions.append(illegal_move_uci)
                    target_frames.append(current_frame_tensor) # Target is the *same* frame

                    episode_transitions += 1
                    illegal_moves_added += 1
                    illegal_move_generated_this_turn = True
                    # --- CRITICAL: DO NOT change board state or current_frame_tensor ---
                    # Skip the legal move section for this iteration
                # else:
                    # No pseudo-legal-but-illegal moves found in this state.
                    # Fall through to generating a legal move.
                    pass

            # --- <<< REGULAR LEGAL MOVE LOGIC >>> ---
            # Only execute if we didn't just generate an illegal move
            if not illegal_move_generated_this_turn:
                if not legal_moves:
                    # This should only happen if the game ended exactly before the illegal check,
                    # or if board.is_game_over() was false but no legal moves exist (stalemate/checkmate missed?)
                    print(f"Warning: No legal moves found unexpectedly in episode {epi_num}. FEN: {board.fen()}")
                    break

                chosen_move = random.choice(legal_moves)
                chosen_move_uci = chosen_move.uci()

                # --- Execute Legal Move ---
                board.push(chosen_move)

                # --- Render *New* State ---
                try:
                    new_svg = chess.svg.board(board=board, size=IMAGE_SIZE)
                    new_pil = svg_to_pil(new_svg)
                    if new_pil is None:
                         print(f"ERROR: Render failed after legal move {chosen_move_uci} ep {epi_num}. Stopping.")
                         game_result = "RenderError"
                         break
                    new_frame_tensor = pil_to_normalized_tensor(new_pil)
                except Exception as render_err:
                     print(f"FATAL: Render error after legal move {chosen_move_uci} ep {epi_num}: {render_err}. Stopping.")
                     game_result = "RenderError"
                     break

                # --- Store the Legal Transition ---
                previous_frames.append(current_frame_tensor)
                actions.append(chosen_move_uci)
                target_frames.append(new_frame_tensor)

                # --- Update for next iteration ---
                current_frame_tensor = new_frame_tensor
                episode_transitions += 1
        # --- End of while loop ---

        if game_result == "*":
            if board.is_game_over(claim_draw=True):
                game_result = board.result(claim_draw=True)
            else:
                 game_result = "Incomplete" # Or more specific if known

        previous_frames.append(current_frame_tensor)
        actions.append(FINAL_ACTION_TOKEN)
        target_frames.append(G)
        episode_transitions += 1

        try:
            stacked_prev_frames = torch.stack(previous_frames).cpu()
            stacked_target_frames = torch.stack(target_frames).cpu()
            assert stacked_prev_frames.shape[0] == len(actions) == stacked_target_frames.shape[0]
            assert stacked_prev_frames.shape[1:] == FRAME_SIZE
        except Exception as stack_err:
            print(f"Error stacking tensors ep {epi_num}: {stack_err}. Skipping save.")
            continue

        episode_dict = {
            'previous_frames': stacked_prev_frames,
            'actions': actions,
            'target_frames': stacked_target_frames,
            'game': GAME_NAME,
            'result': game_result,
            'generation_method': 'random_with_pseudo_illegal', # Updated method name
            'num_transitions': episode_transitions
        }

        filename = f"{GAME_NAME}_epi{epi_num}.pth" 
        save_path = os.path.join(save_dir_abs, filename)
        try:
            torch.save(episode_dict, save_path)
        except Exception as save_err:
             print(f"Error saving episode {epi_num} to {save_path}: {save_err}")

        total_transitions += episode_transitions

    print(f"\n✅ Generated total {total_transitions} transitions across {num_episodes} processed episodes.")
    print(f"   (Including {illegal_moves_added} injected pseudo-illegal move attempts)")
    return save_dir_abs

# --- [4] Visualization Function (Unchanged, but will show identical frames for illegal moves) ---
# (The existing visualize_episode_data function will work correctly)
def visualize_episode_data(episode_dict, episode_filename):
    """Displays previous frame, action, and target frame for each transition."""
    print(f"\n--- Visualizing Episode: {episode_filename} ---")
    print(f"    Game Result: {episode_dict.get('result', 'N/A')}")
    print(f"    Total Transitions: {episode_dict.get('num_transitions', 'N/A')}")
    print(f"    Generation Method: {episode_dict.get('generation_method', 'N/A')}") # Will show new method

    prev_frames = episode_dict['previous_frames'] # Tensor [N, H, W, C]
    actions = episode_dict['actions']             # List [N]
    target_frames = episode_dict['target_frames'] # Tensor [N, H, W, C]
    num_transitions = len(actions)

    if prev_frames.shape[0] != num_transitions or target_frames.shape[0] != num_transitions:
        print("Error: Mismatch between number of actions and number of frames in tensors.")
        return

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(num_transitions):
        #print(f"  Displaying step {i+1}/{num_transitions}...") # Less verbose
        prev_frame_tensor = prev_frames[i]
        action = actions[i]
        target_frame_tensor = target_frames[i]

        is_illegal_attempt = torch.equal(prev_frame_tensor, target_frame_tensor) and action != GAME_NAME and action != FINAL_ACTION_TOKEN

        try:
            prev_img = tensor_to_pil(prev_frame_tensor)
            target_img = tensor_to_pil(target_frame_tensor)
        except Exception as e:
            print(f"Error converting tensors to PIL images at step {i}: {e}")
            prev_img = Image.new('RGB', (FRAME_SIZE[1], FRAME_SIZE[0]), (0, 0, 0))
            target_img = Image.new('RGB', (FRAME_SIZE[1], FRAME_SIZE[0]), (0, 0, 0))

        ax[0].clear()
        ax[1].clear()

        ax[0].imshow(prev_img)
        ax[0].set_title(f"Step {i+1}: Previous Frame")
        ax[0].axis("off")

        ax[1].imshow(target_img)
        title_suffix = " (Illegal Attempt)" if is_illegal_attempt else ""
        ax[1].set_title(f"Action: '{action}'{title_suffix} -> Target")
        ax[1].axis("off")

        fig.suptitle(f"Episode: {episode_filename} - Transition {i+1}/{num_transitions}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.canvas.draw_idle()
        pause_duration = 1.2 if is_illegal_attempt else 0.8 # Pause longer on illegal moves
        plt.pause(pause_duration)

        # Optional: Ask to continue
        # if i % 10 == 0 and i > 0: # Ask every 10 steps
        #     cont = input("Show next 10 steps? (Y/n): ").lower()
        #     if cont == 'n': break

    plt.ioff()
    plt.close(fig)
    print(f"--- Finished visualizing {episode_filename} ---")


# --- [5] Main Execution ---
if __name__ == "__main__":

    print("\n--- Generating Chess Dataset with Random Legal & Pseudo-Illegal Moves ---")

    # --- Matplotlib Backend Check ---
    can_visualize = False
    try:
        import matplotlib
        # Prefer interactive backends
        # Removed MacOSX as it often causes issues if not configured right
        backends_to_try = ['TkAgg', 'Qt5Agg', 'QtAgg'] # Added QtAgg
        used_backend = None
        print("Checking available interactive Matplotlib backends...")
        for backend in backends_to_try:
            try:
                matplotlib.use(backend, force=True) # Force attempt
                # Test if it *really* works
                fig_test = plt.figure()
                plt.close(fig_test)
                print(f"Successfully using Matplotlib backend: {backend}")
                used_backend = backend
                can_visualize = True
                break # Stop on first success
            except Exception as e:
                 # print(f"  Backend '{backend}' failed: {e}") # More verbose debug
                 print(f"  Matplotlib backend '{backend}' not available or failed.")
                 matplotlib.use('Agg') # Reset to Agg in case backend switch failed partially
                 continue

        if not used_backend:
            matplotlib.use('Agg') # Explicitly set fallback
            print("No interactive Matplotlib backend found. Using 'Agg' (non-interactive).")
            print("Visualization will not show plots in real-time.")
            can_visualize = False # Set to false as real-time viewing won't work

    except ImportError:
        print("\nWARNING: matplotlib is not installed. Visualization will be skipped.")
        print("Install using: pip install matplotlib")
        can_visualize = False
    except Exception as e:
         print(f"\nERROR during Matplotlib setup: {e}")
         print("Visualization might be disabled.")
         can_visualize = False

    # --- Generate the Dataset ---
    dataset_dir = generate_chess_episodes_random(
        num_episodes=N_EPISODES_DATASET,
    )

    # --- Post-Generation Check and Visualization ---
    if dataset_dir and os.path.exists(dataset_dir):
        print(f"\nDataset generation finished. Episodes saved in: {dataset_dir}")
        try:
            print("\n--- Checking Generated Files (First Episode) ---")
            # Update filename pattern for finding files
            episode_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".pth") and f.startswith(f"{GAME_NAME}")])
            if not episode_files:
                print("No generated episode '.pth' files found (check filename pattern and path).")
            else:
                 first_episode_path = os.path.join(dataset_dir, episode_files[0])
                 print(f"Loading first file: {first_episode_path}")
                 episode_data = torch.load(first_episode_path, map_location=torch.device('cpu'))

                 # Print metadata and shapes (check generation_method)
                 print("  Metadata:")
                 print(f"    Game: {episode_data.get('game', 'N/A')}")
                 print(f"    Result: {episode_data.get('result', 'N/A')}")
                 print(f"    Generation Method: {episode_data.get('generation_method', 'N/A')}")
                 print(f"    Num Transitions: {episode_data.get('num_transitions', 'N/A')}")
                 print("  Data Structure:")
                 # ... (shape printing remains the same) ...
                 if 'previous_frames' in episode_data: print(f"    previous_frames: Tensor, Shape={episode_data['previous_frames'].shape}, Dtype={episode_data['previous_frames'].dtype}")
                 else: print("    previous_frames: MISSING")
                 if 'actions' in episode_data: print(f"    actions: List, Length={len(episode_data['actions'])}")
                 else: print("    actions: MISSING")
                 if 'target_frames' in episode_data: print(f"    target_frames: Tensor, Shape={episode_data['target_frames'].shape}, Dtype={episode_data['target_frames'].dtype}")
                 else: print("    target_frames: MISSING")


                 # Basic sanity check (remains the same)
                 num_trans_meta = episode_data.get('num_transitions', -1)
                 num_actions_list = len(episode_data.get('actions', []))
                 num_prev_frames = episode_data.get('previous_frames', torch.empty(0)).shape[0]
                 num_target_frames = episode_data.get('target_frames', torch.empty(0)).shape[0]
                 if num_trans_meta == num_actions_list == num_prev_frames == num_target_frames: print("  Basic shape/count check: PASSED")
                 else: print(f"  Basic shape/count check: FAILED (Meta: {num_trans_meta}, Actions: {num_actions_list}, Prev: {num_prev_frames}, Target: {num_target_frames})")


                 # --- Visualization Prompt ---
                 # Only prompt if interactive backend worked and not in CI
                 if can_visualize and matplotlib.get_backend() != 'Agg' and not os.environ.get('CI'):
                     visualize_choice = input(f"\nDo you want to visualize an episode? (y/N): ").lower()
                     if visualize_choice == 'y':
                         print("\nAvailable episodes:")
                         for idx, fname in enumerate(episode_files): print(f"  {idx+1}: {fname}")
                         while True:
                             try:
                                 choice_idx_str = input(f"Enter number (1-{len(episode_files)}) or Enter to skip: ")
                                 if not choice_idx_str: break
                                 choice_idx = int(choice_idx_str) - 1
                                 if 0 <= choice_idx < len(episode_files):
                                     chosen_file = episode_files[choice_idx]
                                     chosen_path = os.path.join(dataset_dir, chosen_file)
                                     print(f"Loading {chosen_file}...")
                                     vis_episode_data = torch.load(chosen_path, map_location=torch.device('cpu'))
                                     visualize_episode_data(vis_episode_data, chosen_file)
                                     another = input("Visualize another? (y/N): ").lower()
                                     if another != 'y': break
                                     else: # Show list again
                                          print("\nAvailable episodes:")
                                          for idx, fname in enumerate(episode_files): print(f"  {idx+1}: {fname}")
                                 else: print("Invalid number.")
                             except ValueError: print("Invalid input.")
                             except KeyboardInterrupt: print("\nVisualization interrupted."); break
                             except Exception as e: print(f"\nViz Error: {e}"); break # Exit loop on error
                 elif can_visualize and matplotlib.get_backend() == 'Agg':
                      print("\nMatplotlib backend is 'Agg' (non-interactive). Skipping real-time visualization prompt.")
                 elif not can_visualize:
                      print("\nVisualization skipped (matplotlib issue or not installed).")

        except Exception as e:
            print(f"\nError during generated file check or visualization prompt: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\nDataset generation may have failed or the output directory was not found.")
        print("Check console output for errors (e.g., cairosvg setup, file permissions).")

    print("\n--- Script Finished ---")
