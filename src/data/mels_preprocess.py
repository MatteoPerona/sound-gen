import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import numpy as np
from tqdm import tqdm

# IMPORTANT: RUN FROM /SRC NOT SRC/DATA


input_dir = "raw\\mels"      # Directory with .npy mel spectrograms
output_dir = "clean\\mels" # Output directory for fixed-length mels
os.makedirs(output_dir, exist_ok=True)

target_duration = AUDIO_IN_SECONDS   # seconds
hop_length = HOP_LENGTH      # Use the same hop_length as in your mel calculation
sr = SAMPLE_RATE            # Use the same sample rate as in your mel calculation

# Calculate target number of frames for 30s
target_frames = int(np.ceil(target_duration * sr / hop_length))

# Collect all .npy file paths first
all_files = []
for root, _, files in os.walk(input_dir):
    for fname in files:
        if fname.endswith(".npy"):
            all_files.append(os.path.join(root, fname))

# Now process with a single tqdm bar
for in_path in tqdm(all_files, desc=f"Processing melspecs with {target_duration}s duration"):
    mel = np.load(in_path)
    n_mels, n_frames = mel.shape

    if n_frames > target_frames:
        mel_fixed = mel[:, :target_frames]
    elif n_frames < target_frames:
        pad_width = target_frames - n_frames
        mel_fixed = np.pad(mel, ((0,0), (0, pad_width)), mode='constant')
    else:
        mel_fixed = mel

    # Save to output directory, preserving subfolder structure
    rel_path = os.path.relpath(in_path, input_dir)
    out_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mel_fixed)