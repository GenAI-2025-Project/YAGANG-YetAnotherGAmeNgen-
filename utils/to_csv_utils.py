# -*- coding: utf-8 -*-
import torch
import os
import glob
from tqdm import tqdm
from collections import Counter
import sys
import csv

# --- Configuration ---
# <<< UPDATE THESE PATHS/PREFIXES >>>
DATA_DIR = "./dataset/"             # Directory containing your .pth episode files
EPISODE_FILE_PREFIX = "<game" # <<< SET A SPECIFIC PREFIX for the game/episodes
                                      # Example: "game_level1_"
                                      # If empty (""), NO CSV will be generated.
                                      # This prefix also determines the CSV filename.
OUTPUT_CSV_DIR = "./actions_csv/" # Directory to save the CSV file
ACTION_DELIMITER = ";" # Delimiter to join multiple actions within a single cell

# Key names to check for the action list within the loaded dictionary
ACTION_KEYS = ['actions', 'a']

# --- Script Logic ---

def analyze_action_statistics(data_dir, episode_prefix, output_csv_dir, action_delimiter):
    """
    Analyzes action statistics from .pth files and saves a summary CSV
    (filename, all_actions_joined) if a specific prefix is provided.

    Args:
        data_dir (str): Path to the directory containing dataset files.
        episode_prefix (str): Prefix to filter filenames. If non-empty,
                              a summary CSV will be saved.
        output_csv_dir (str): Directory where the CSV file will be saved.
        action_delimiter (str): String used to join actions in the CSV cell.

    Returns:
        tuple: (total_actions, unique_actions_count, unique_actions_set)
               Returns (0, 0, set()) if no valid data is found.
    """
    print(f"--- Action Statistics Analysis ---")
    print(f"Scanning Directory: '{os.path.abspath(data_dir)}'")
    print(f"Filtering by Prefix: '{episode_prefix}'")
    if episode_prefix:
        print(f"CSV Output Directory: '{os.path.abspath(output_csv_dir)}'")
        csv_filename_preview = f"{episode_prefix}_all_actions_per_file.csv"
        print(f"CSV Filename Pattern: '{csv_filename_preview}'")
        print(f"CSV Format: Filename, Actions (joined by '{action_delimiter}')")
    else:
        print("CSV Output: Disabled (EPISODE_FILE_PREFIX is empty)")


    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return 0, 0, set()

    file_pattern = os.path.join(data_dir, f"{episode_prefix}*.pth")
    files_to_process = sorted(glob.glob(file_pattern))

    if not files_to_process:
        print(f"Warning: No files found matching pattern: '{file_pattern}'")
        print("Check DATA_DIR and EPISODE_FILE_PREFIX.")
        return 0, 0, set()

    print(f"Found {len(files_to_process)} files matching the pattern.")

    # Store tuples of (filename, list_of_actions)
    file_action_data = []
    # Also collect all individual actions for overall statistics
    all_individual_actions = []
    files_processed = 0
    files_skipped_errors = 0
    files_skipped_no_actions = 0

    for fpath in tqdm(files_to_process, desc="Processing files"):
        filename = os.path.basename(fpath)
        try:
            if os.path.getsize(fpath) < 100:
                 files_skipped_errors += 1
                 continue

            data = torch.load(fpath, map_location='cpu')

            if not isinstance(data, dict):
                tqdm.write(f"Warning: File '{filename}' is not a dictionary. Skipping.")
                files_skipped_errors += 1
                continue

            actions = None
            for key in ACTION_KEYS:
                if key in data:
                    actions = data[key]
                    break

            if actions is None:
                tqdm.write(f"Warning: No action key found in '{filename}'. Skipping.")
                files_skipped_no_actions += 1
                continue

            if not isinstance(actions, list):
                tqdm.write(f"Warning: 'actions' data is not a list in '{filename}'. Skipping.")
                files_skipped_errors += 1
                continue

            # Filter for valid string actions BEFORE adding to lists
            valid_actions_in_file = [a for a in actions if isinstance(a, str) and a.strip()]

            if valid_actions_in_file: # Only process if there are valid actions
                file_action_data.append((filename, valid_actions_in_file))
                all_individual_actions.extend(valid_actions_in_file) # Add to overall list
                files_processed += 1
            else:
                # Skipped because although an action list might exist, it was empty or contained invalid data
                files_skipped_no_actions += 1


        except Exception as e:
            tqdm.write(f"Error processing file '{filename}': {type(e).__name__} - {e}")
            files_skipped_errors += 1
            continue

    print("\n--- Analysis Complete ---")

    if not file_action_data: # Check if we collected any file data for the CSV
        print("No files with valid actions found.")
        print(f"Files Scanned: {len(files_to_process)}")
        print(f"  Skipped (Errors/Format): {files_skipped_errors}")
        print(f"  Skipped (No/Empty/Invalid Actions): {files_skipped_no_actions}")
        if episode_prefix:
             print("\nNo CSV file generated as no actionable data was found.")
        return 0, 0, set()

    # --- Calculate Overall Statistics ---
    total_actions = len(all_individual_actions)
    unique_actions_set = set(all_individual_actions)
    unique_actions_count = len(unique_actions_set)

    print(f"Files Successfully Processed (containing valid actions): {files_processed}")
    print(f"Files Skipped (Errors/Format Issues): {files_skipped_errors}")
    print(f"Files Skipped (No Actions Key or Empty/Invalid List): {files_skipped_no_actions}")
    print("-" * 20)
    print(f"Total Individual Actions Found (across all files): {total_actions}")
    print(f"Unique Actions Found (across all files): {unique_actions_count}")
    print("-" * 20)

    # --- CSV Saving Logic ---
    if episode_prefix and file_action_data: # Check prefix AND if we have data to write
        try:
            os.makedirs(output_csv_dir, exist_ok=True)
            # Updated filename for clarity
            csv_filename = f"{episode_prefix}_all_actions_per_file.csv"
            output_csv_path = os.path.join(output_csv_dir, csv_filename)

            print(f"Saving action summary for {len(file_action_data)} files to: {output_csv_path}")

            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write Header
                csv_writer.writerow(['Filename', 'Actions']) # Single column for all actions

                # Write Data Rows
                for filename, actions_list in file_action_data:
                    # Join the list of actions into a single string
                    actions_string = action_delimiter.join(actions_list)
                    csv_writer.writerow([filename, actions_string])

            print("CSV file saved successfully.")

        except Exception as e:
            print(f"\nError saving CSV file '{output_csv_path}': {type(e).__name__} - {e}")
    elif episode_prefix and not file_action_data:
        print("\nNo CSV file generated as no actionable data was found.")
    # ----------------------


    # Optional: Print unique actions list (based on overall statistics)
    if unique_actions_count > 0 and unique_actions_count <= 100:
        print("\nUnique Actions List (Overall):")
        for action in sorted(list(unique_actions_set)):
            print(f"  - {action}")
    elif unique_actions_count > 100:
        print(f"\nUnique Actions List (Overall, {unique_actions_count}) is too long to display here.")

    return total_actions, unique_actions_count, unique_actions_set

if __name__ == "__main__":
    try:
        if EPISODE_FILE_PREFIX:
            os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create output directory '{OUTPUT_CSV_DIR}': {e}")

    total_count, unique_count, unique_set = analyze_action_statistics(
        DATA_DIR,
        EPISODE_FILE_PREFIX,
        OUTPUT_CSV_DIR,
        ACTION_DELIMITER
    )

    # The summary print remains the same, reporting overall action counts
    if total_count > 0:
        print("\nSummary (Overall Across Files):")
        print(f"  Total individual actions: {total_count}")
        print(f"  Unique actions: {unique_count}")
    else:
        print("\nNo actionable data found based on the configuration.")
