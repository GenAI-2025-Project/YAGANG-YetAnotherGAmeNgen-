import torch
import os
import glob
from tqdm import tqdm
import argparse
import traceback

def count_transitions_in_files(data_dir, episode_prefix):
    """
    Counts the number of transitions stored in .pth files matching a prefix.

    Args:
        data_dir (str): The directory containing the dataset files.
        episode_prefix (str): The prefix for the episode files to scan.
    """
    print(f"[Counter] Scanning directory: '{data_dir}' for files starting with: '{episode_prefix}'")

    # Construct the search pattern
    pattern = os.path.join(data_dir, f"{episode_prefix}*.pth")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[Counter] No files found matching pattern: {pattern}")
        return

    print(f"[Counter] Found {len(files)} files matching the pattern.")

    total_transitions_across_files = 0
    file_transition_counts = {}

    # Use tqdm for progress visualization
    iterator = tqdm(files, desc="[Counter] Processing files", leave=False)

    for fpath in iterator:
        try:
            # Basic check if file exists and has some content
            if not os.path.exists(fpath) or os.path.getsize(fpath) < 100:
                print(f"[Counter] Skipping small/missing file: {os.path.basename(fpath)}")
                continue

            # Load data from the file onto CPU to avoid GPU memory usage
            data = torch.load(fpath, map_location='cpu')

            # --- Data Structure Handling ---
            # Check if it's a dictionary and try to find the frame tensor.
            # Handles both {'p': tensor, ...} and {'previous_frames': tensor, ...}
            frames_tensor = None
            if isinstance(data, dict):
                if 'p' in data and isinstance(data['p'], torch.Tensor):
                    frames_tensor = data['p']
                elif 'previous_frames' in data and isinstance(data['previous_frames'], torch.Tensor):
                    frames_tensor = data['previous_frames']
                # Add more potential keys if your structure varies
            else:
                 # If it's not a dict, maybe it's just the tensor directly? Less likely.
                 # Or maybe an older format. Add handling if needed.
                 print(f"[Counter] Warning: Unexpected data type ({type(data)}) in file: {os.path.basename(fpath)}. Skipping.")
                 continue

            # --- Validation and Counting ---
            if frames_tensor is None:
                print(f"[Counter] Warning: Could not find frame tensor ('p' or 'previous_frames') in file: {os.path.basename(fpath)}. Skipping.")
                continue

            # Check tensor dimensions (expecting Batch, C, H, W or Batch, H, W, C)
            if frames_tensor.ndim < 2: # Need at least a batch dimension
                 print(f"[Counter] Warning: Frame tensor in {os.path.basename(fpath)} has unexpected dimensions ({frames_tensor.ndim}). Skipping.")
                 continue

            # The number of transitions is the size of the first dimension
            num_transitions = frames_tensor.shape[0]

            file_transition_counts[os.path.basename(fpath)] = num_transitions
            total_transitions_across_files += num_transitions

        except FileNotFoundError:
            print(f"[Counter] Error: File not found during processing (should not happen with glob): {os.path.basename(fpath)}")
        except (pickle.UnpicklingError, EOFError, RuntimeError) as load_err: # Catch common torch.load errors
             print(f"[Counter] Error loading file {os.path.basename(fpath)}: {load_err}. Skipping.")
        except KeyError as key_err:
             print(f"[Counter] Error accessing data (missing key?) in {os.path.basename(fpath)}: {key_err}. Skipping.")
        except Exception as e:
            # Catch any other unexpected errors during processing
            print(f"\n[Counter] Unexpected Error processing file: {os.path.basename(fpath)}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            # traceback.print_exc() # Uncomment for full traceback if needed
            print("-" * 20)
            continue # Skip to the next file

    # Close the tqdm progress bar
    iterator.close()
    print("\n[Counter] Finished processing files.")

    # Print results
    if not file_transition_counts:
        print("[Counter] No valid transitions found in any matching files.")
    else:
        print("\n--- Transition Counts Per File ---")
        # Sort by filename for consistent output
        for filename, count in sorted(file_transition_counts.items()):
            print(f"  {filename}: {count} transitions")

        print("\n--- Summary ---")
        print(f"Total transitions found across {len(file_transition_counts)} valid files: {total_transitions_across_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count transitions in dataset .pth files.")
    parser.add_argument("data_dir", type=str, help="Directory containing the dataset files.")
    parser.add_argument("episode_prefix", type=str, help="Prefix of the episode files to scan (e.g., 'episode_').")

    args = parser.parse_args()

    count_transitions_in_files(args.data_dir, args.episode_prefix)
