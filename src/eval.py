import torch
import numpy as np
from models.gan import Generator
from config import N_MELS, AUDIO_LENGTH
import soundfile as sf

# ---- CONFIG ----
CHECKPOINT_PATH = "checkpoints/generator_epoch_0.pt"  # Update this
MEL_SPEC_PATH = "data/clean/mels/00/7400.npy"      # Update this
OUTPUT_WAV_PATH = "data/generated_audio.wav"
SAMPLE_RATE = 22050  # Update if needed

# ---- LOAD GENERATOR ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(n_mels=N_MELS, audio_length=AUDIO_LENGTH).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
generator.load_state_dict(checkpoint['model_state_dict'])
generator.eval()

# ---- LOAD MEL-SPECTROGRAM ----
mel_spec_np = np.load(MEL_SPEC_PATH)  # shape: (n_mels, time)
if mel_spec_np.ndim == 2:
    mel_spec_tensor = torch.tensor(mel_spec_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
elif mel_spec_np.ndim == 3:
    mel_spec_tensor = torch.tensor(mel_spec_np, dtype=torch.float32).unsqueeze(0)  # (1, 1, n_mels, time)
else:
    raise ValueError("Mel spectrogram must be 2D or 3D numpy array")
mel_spec_tensor = mel_spec_tensor.to(device)

# ---- GENERATE AUDIO ----
with torch.no_grad():
    generated_audio = generator(mel_spec_tensor)  # (1, audio_length)
audio_np = generated_audio.squeeze().cpu().numpy()
print(f"Generated audio shape: {audio_np.shape}")

# ---- SAVE AUDIO ----
sf.write(OUTPUT_WAV_PATH, audio_np, SAMPLE_RATE)
print(f"Generated audio saved to {OUTPUT_WAV_PATH}")