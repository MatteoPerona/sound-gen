import os
import numpy as np
from tqdm import tqdm

input_dir = "moodtheme-melspecs"      # Directory with .npy mel spectrograms
output_dir = "moodtheme-melspecs-30s" # Output directory for fixed-length mels
os.makedirs(output_dir, exist_ok=True)

target_duration = 30  # seconds
hop_length = 512      # Use the same hop_length as in your mel calculation
sr = 22050            # Use the same sample rate as in your mel calculation

# Calculate target number of frames for 30s
target_frames = int(np.ceil(target_duration * sr / hop_length))

for root, _, files in os.walk(input_dir):
    for fname in tqdm(files, desc="Processing melspecs"):
        if not fname.endswith(".npy"):
            continue
        in_path = os.path.join(root, fname)
        mel = np.load(in_path)
        # mel shape: (n_mels, time_frames)
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