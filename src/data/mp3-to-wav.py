import os
import librosa
import numpy as np
from tqdm import tqdm

base_dir = "moodtheme-mp3"
max_length = 0
max_file = ""

# Gather all mp3 file paths first
mp3_files = []
for nn in range(100):
    subdir = os.path.join(base_dir, f"{nn:02d}")
    if not os.path.isdir(subdir):
        continue
    for fname in os.listdir(subdir):
        if fname.lower().endswith(".mp3"):
            mp3_files.append(os.path.join(subdir, fname))

# Process with tqdm progress bar
for fpath in tqdm(mp3_files, desc="Processing MP3s"):
    try:
        y, sr = librosa.load(fpath, sr=None, mono=True)
        if len(y) > max_length:
            max_length = len(y)
            max_file = fpath
    except Exception as e:
        print(f"Error processing {fpath}: {e}")

print(f"Maximum length: {max_length} samples")
print(f"File: {max_file}")