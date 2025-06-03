import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


base_dir = "raw/audio"
output_dir = "clean/audio"
os.makedirs(output_dir, exist_ok=True)

target_sr = SAMPLE_RATE  # or any sample rate you want
target_duration = AUDIO_IN_SECONDS  # seconds
target_length = AUDIO_LENGTH
# target_length = target_sr * target_duration 

# Gather all mp3 file paths
mp3_files = []
for nn in range(100):
    subdir = os.path.join(base_dir, f"{nn:02d}")
    if not os.path.isdir(subdir):
        print(f"Skipping non-directory: {subdir}")
        continue
    for fname in os.listdir(subdir):
        if fname.lower().endswith(".mp3"):
            mp3_files.append(os.path.join(subdir, fname))

def process_and_save(fpath):
    try:
        y, sr = librosa.load(fpath, sr=target_sr, mono=True)
        # Split into 15-second chunks
        chunk_length = target_sr * 15  # 15 seconds in samples
        num_chunks = int(np.ceil(len(y) / chunk_length))
        rel_path = os.path.relpath(fpath, base_dir)
        base_out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0])
        for idx in range(num_chunks):
            start = idx * chunk_length
            end = min((idx + 1) * chunk_length, len(y))
            chunk = y[start:end]
            # Pad if last chunk is shorter than 15 seconds
            if len(chunk) < chunk_length:
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)), 'constant')
            out_path = f"{base_out_path}_chunk{idx:03d}.wav"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, chunk, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return False

if __name__ == "__main__":
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_and_save, fpath) for fpath in mp3_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Converting to {AUDIO_IN_SECONDS}s WAVs'):
            result = future.result()
            if result:
                success_count += 1
            else:
                fail_count += 1

    print(f"Done! {success_count} files processed successfully, {fail_count} failed.")

    # print("Converting Wavs to Melspectrograms...")
    # from src.data.clean_wav_to_mel import convert_wavs_to_mels
    # convert_wavs_to_mels()