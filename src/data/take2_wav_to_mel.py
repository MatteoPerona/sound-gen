import os
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


# === Configuration ===
INPUT_ROOT = 'clean/audio'
OUTPUT_ROOT = 'clean/mels'

SAMPLE_RATE = 12000
N_MELS = 96
HOP_LENGTH = 256
WIN_LENGTH = 512
F_MIN = 0
F_MAX = SAMPLE_RATE // 2


def process_file(input_output):
    input_path, output_path = input_output
    try:
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX,
            power=2.0
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, mel_db)

    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")


def collect_files(input_root, output_root):
    file_pairs = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith('.wav'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_root)
                output_file = os.path.splitext(relative_path)[0] + '.npy'
                output_path = os.path.join(output_root, output_file)
                file_pairs.append((input_path, output_path))
    return file_pairs


if __name__ == '__main__':
    file_pairs = collect_files(INPUT_ROOT, OUTPUT_ROOT)
    print(f"📁 Found {len(file_pairs)} WAV files")

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, file_pairs), total=len(file_pairs), desc="Processing"):
            pass
