import os
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = "moodtheme-mp3"
max_length = 0
max_file = ""
max_time = 0.0
durations = []
results = []

# Gather all mp3 file paths first
mp3_files = []
for nn in range(100):
    subdir = os.path.join(base_dir, f"{nn:02d}")
    if not os.path.isdir(subdir):
        continue
    for fname in os.listdir(subdir):
        if fname.lower().endswith(".mp3"):
            mp3_files.append(os.path.join(subdir, fname))

def process_file(fpath):
    try:
        y, sr = librosa.load(fpath, sr=None, mono=True)
        duration = len(y) / sr if sr else 0
        return (fpath, len(y), duration)
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return (fpath, 0, 0)

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, fpath): fpath for fpath in mp3_files}
    for future in tqdm(as_completed(futures), total=len(mp3_files), desc="Processing MP3s"):
        results.append(future.result())

# Extract metrics
for fpath, length, duration in results:
    durations.append(duration)
    if length > max_length:
        max_length = length
        max_file = fpath
        max_time = duration

mean_time = np.mean(durations) if durations else 0
print(f"Maximum length: {max_length} samples")
print(f"Maximum time: {max_time:.2f} seconds")
print(f"Mean time: {mean_time:.2f} seconds")
print(f"File: {max_file}")

# Plot histogram of durations
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of MP3 Durations")
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.grid(True)
plt.show()