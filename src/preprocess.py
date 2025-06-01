import torch
import torchaudio
import numpy as np
from pathlib import Path
import tqdm
import multiprocessing
from functools import partial
from config import AUDIO_LENGTH, SAMPLE_RATE, HOP_LENGTH

def process_audio(audio_path, output_dir, audio_length=AUDIO_LENGTH, sample_rate=SAMPLE_RATE):
    """Process a single audio file to ensure consistent length and format."""
    try:
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
        
        # Ensure consistent length
        if audio.shape[1] > audio_length:
            # If too long, truncate to first segment
            audio = audio[:, :audio_length]
        else:
            # If too short, pad with zeros
            padding = audio_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Create output path maintaining directory structure
        rel_path = audio_path.relative_to(Path("data/raw/audio"))
        audio_output_path = output_dir / "clean" / "audio" / rel_path.parent / f"{rel_path.stem}.wav"
        
        # Create directory
        audio_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed audio
        torchaudio.save(audio_output_path, audio, sample_rate)
        
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False

def process_mel(mel_path, output_dir, audio_length=AUDIO_LENGTH, hop_length=HOP_LENGTH):
    """Process a single mel spectrogram to ensure consistent length."""
    try:
        # Load mel spectrogram
        mel = np.load(mel_path)
        
        # Calculate expected mel length based on audio length and hop length
        expected_mel_length = (audio_length // hop_length) + 1
        
        # Ensure consistent length
        if mel.shape[1] > expected_mel_length:
            # If too long, truncate to first segment
            mel = mel[:, :expected_mel_length]
        else:
            # If too short, pad with zeros
            padding = expected_mel_length - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, padding)))
        
        # Create output path maintaining directory structure
        rel_path = mel_path.relative_to(Path("data/raw/mels"))
        mel_output_path = output_dir / "clean" / "mels" / rel_path.parent / f"{rel_path.stem}.npy"
        
        # Create directory
        mel_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed mel spectrogram
        np.save(mel_output_path, mel)
        
        return True
    except Exception as e:
        print(f"Error processing {mel_path}: {str(e)}")
        return False

def preprocess_dataset(raw_dir="data/raw", output_dir="data"):
    """Preprocess all audio files and mel spectrograms in the dataset."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    
    # Get all audio files
    audio_files = []
    for ext in ['.mp3', '.wav', '.flac']:
        audio_files.extend(list((raw_dir / "audio").glob(f"**/*{ext}")))
    
    # Get all mel spectrograms
    mel_files = list((raw_dir / "mels").glob("**/*.npy"))
    
    if not audio_files and not mel_files:
        raise ValueError(f"No audio files or mel spectrograms found in {raw_dir}")
    
    print(f"Found {len(audio_files)} audio files and {len(mel_files)} mel spectrograms")
    print(f"Processing audio to {AUDIO_LENGTH/SAMPLE_RATE:.1f} seconds ({AUDIO_LENGTH} samples) at {SAMPLE_RATE}Hz")
    print(f"Processing mel spectrograms to {(AUDIO_LENGTH // HOP_LENGTH) + 1} frames")
    
    # Process audio files in parallel
    if audio_files:
        process_audio_func = partial(process_audio, 
                                   output_dir=output_dir,
                                   audio_length=AUDIO_LENGTH,
                                   sample_rate=SAMPLE_RATE)
        
        with multiprocessing.Pool() as pool:
            audio_results = list(tqdm.tqdm(
                pool.imap(process_audio_func, audio_files),
                total=len(audio_files),
                desc="Processing audio files"
            ))
        
        # Print audio statistics
        successful_audio = sum(audio_results)
        print(f"Successfully processed {successful_audio}/{len(audio_files)} audio files")
    
    # Process mel spectrograms in parallel
    if mel_files:
        process_mel_func = partial(process_mel,
                                 output_dir=output_dir,
                                 audio_length=AUDIO_LENGTH,
                                 hop_length=HOP_LENGTH)
        
        with multiprocessing.Pool() as pool:
            mel_results = list(tqdm.tqdm(
                pool.imap(process_mel_func, mel_files),
                total=len(mel_files),
                desc="Processing mel spectrograms"
            ))
        
        # Print mel statistics
        successful_mels = sum(mel_results)
        print(f"Successfully processed {successful_mels}/{len(mel_files)} mel spectrograms")
    
    # Print output structure
    print("\nOutput structure:")
    print(f"Clean audio: {output_dir}/clean/audio/")
    print(f"Clean mel spectrograms: {output_dir}/clean/mels/")

if __name__ == "__main__":
    # Create output directories
    Path("data/clean/audio").mkdir(parents=True, exist_ok=True)
    Path("data/clean/mels").mkdir(parents=True, exist_ok=True)
    
    # Preprocess dataset
    preprocess_dataset() 