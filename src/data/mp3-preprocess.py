import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

base_dir = "moodtheme-mp3"
output_dir = "moodtheme-wav-30s"
os.makedirs(output_dir, exist_ok=True)

target_sr = 22050  # or any sample rate you want
target_duration = 30  # seconds
target_length = target_sr * target_duration

# Gather all mp3 file paths
mp3_files = []
for nn in range(100):
    subdir = os.path.join(base_dir, f"{nn:02d}")
    if not os.path.isdir(subdir):
        continue
    for fname in os.listdir(subdir):
        if fname.lower().endswith(".mp3"):
            mp3_files.append(os.path.join(subdir, fname))

def process_and_save(fpath):
    try:
        y, sr = librosa.load(fpath, sr=target_sr, mono=True)
        # Truncate or pad to 30 seconds
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        # Prepare output path
        rel_path = os.path.relpath(fpath, base_dir)
        out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return False

if __name__ == "__main__":
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_and_save, fpath) for fpath in mp3_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting to 30s WAVs"):
            result = future.result()
            if result:
                success_count += 1
            else:
                fail_count += 1

    print(f"Done! {success_count} files processed successfully, {fail_count} failed.")