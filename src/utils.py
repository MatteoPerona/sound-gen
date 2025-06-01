import torch
import torchaudio
import librosa
import numpy as np
import wandb
from pathlib import Path

def load_audio(file_path, sample_rate=22050):
    """Load audio file and convert to mono."""
    audio, sr = torchaudio.load(file_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    return audio.squeeze()

# def 2(audio, sample_rate=22050, n_mels=80, hop_length=256, win_length=1024):
#     """Convert audio to mel spectrogram."""
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=sample_rate,
#         n_mels=n_mels,
#         hop_length=hop_length,
#         win_length=win_length
#     )
#     mel_spec = mel_transform(audio)
#     return mel_spec

def save_audio(audio, file_path, sample_rate=22050):
    """Save audio tensor to file."""
    torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)

def log_audio_samples(gen_audio, mel_spec, step, sample_rate=22050):
    """Log audio samples and spectrograms to wandb."""
    # Convert tensors to numpy arrays
    gen_audio_np = gen_audio.detach().cpu().numpy()
    mel_spec_np = mel_spec.detach().cpu().numpy()
    
    # Log audio
    wandb.log({
        "generated_audio": wandb.Audio(
            gen_audio_np,
            sample_rate=sample_rate,
            caption=f"Generated Audio (Step {step})"
        ),
        "mel_spectrogram": wandb.Image(
            mel_spec_np,
            caption=f"Mel Spectrogram (Step {step})"
        )
    })

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'] 