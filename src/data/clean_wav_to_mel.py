import os
import numpy as np
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# CONFIG
input_dir = "clean/audio"      # Directory with .wav files
output_dir = "clean/mels"      # Output directory for mel spectrograms
os.makedirs(output_dir, exist_ok=True)

def process_wav(in_path):
    try:
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=WIN_LENGTH,         # Use your config's FFT/window size
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=F_MIN,               # Set min frequency (e.g., 0 or 20)
            fmax=F_MAX                # Set max frequency (e.g., sr//2 or 8000/11025)
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Save to output directory, preserving subfolder structure
        rel_path = os.path.relpath(in_path, input_dir)
        out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mel_db)
        return True
    except Exception as e:
        print(f"Error processing {in_path}: {e}")
        return False

def convert_wavs_to_mels():
    wav_files = []
    # Search for wavs in numbered subdirectories like mp3_preprocess
    for nn in range(100):
        subdir = os.path.join(input_dir, f"{nn:02d}")
        if not os.path.isdir(subdir):
            print(f"Skipping non-directory: {subdir}")
            continue
        for fname in os.listdir(subdir):
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(subdir, fname))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_wav, fpath) for fpath in wav_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting WAVs to mels"):
            pass

if __name__ == "__main__":
    convert_wavs_to_mels()