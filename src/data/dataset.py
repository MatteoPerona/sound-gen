import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torchaudio
from config import AUDIO_LENGTH, SAMPLE_RATE

class MelSpectrogramDataset(Dataset):
    def __init__(self, mels_dir="data/mels", audio_dir="data/audio"):
        self.mels_dir = Path(mels_dir)
        self.audio_dir = Path(audio_dir)
        
        # Get all mel spectrogram files
        self.mel_files = []
        for mel_dir in sorted(self.mels_dir.glob("*")):
            if mel_dir.is_dir():
                self.mel_files.extend(list(mel_dir.glob("*.npy")))
        
        if not self.mel_files:
            raise ValueError(f"No .npy files found in {mels_dir} and its subdirectories")
        
        print(f"Found {len(self.mel_files)} mel spectrogram files")
        print(f"Expected audio length: {AUDIO_LENGTH} samples ({AUDIO_LENGTH/SAMPLE_RATE:.1f} seconds) at {SAMPLE_RATE}Hz")
    
    def __len__(self):
        return len(self.mel_files)
    
    def __getitem__(self, idx):
        mel_path = self.mel_files[idx]
        
        # Get corresponding audio path
        rel_path = mel_path.relative_to(self.mels_dir)
        audio_path = self.audio_dir / rel_path.parent / f"{rel_path.stem}.wav"
        
        if not audio_path.exists():
            raise ValueError(f"No corresponding audio file found for {mel_path}")
        
        # Load mel spectrogram
        mel_spec = np.load(mel_path)
        mel_spec = torch.from_numpy(mel_spec).float()
        
        # Add channel dimension if not present
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        # Normalize to [-1, 1] if not already normalized
        if mel_spec.max() > 1 or mel_spec.min() < -1:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Load audio file
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Verify sample rate
        if sr != SAMPLE_RATE:
            raise ValueError(f"Sample rate mismatch: expected {SAMPLE_RATE}, got {sr}")
        
        # Verify audio length
        if audio.shape[1] != AUDIO_LENGTH:
            raise ValueError(f"Audio length mismatch: expected {AUDIO_LENGTH}, got {audio.shape[1]}")
        
        return mel_spec, audio.squeeze() 