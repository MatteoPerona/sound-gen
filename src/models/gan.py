import torch
import torch.nn as nn
import torch.nn.functional as F
from config import AUDIO_LENGTH, N_MELS

class Generator(nn.Module):
    def __init__(self, n_mels=N_MELS, audio_length=AUDIO_LENGTH):
        super(Generator, self).__init__()
        
        # Initial processing of mel spectrogram
        self.mel_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Upsampling blocks to generate audio
        self.upsampling_blocks = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second upsampling block
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third upsampling block
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth upsampling block
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer to get audio waveform
            nn.Conv1d(16, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        
        # Project mel spectrogram to initial audio length
        self.mel_projection = nn.Linear(n_mels, audio_length // 16)
    
    def forward(self, mel_spec):
        # Input shape: (batch_size, 1, n_mels, time_steps)
        batch_size = mel_spec.shape[0]
        
        # Encode mel spectrogram
        x = self.mel_encoder(mel_spec)  # (batch_size, 128, n_mels, time_steps)
        
        # Project to initial audio length
        x = x.permute(0, 1, 3, 2)  # (batch_size, 128, time_steps, n_mels)
        x = x.reshape(batch_size, 128, -1)  # (batch_size, 128, time_steps * n_mels)
        x = self.mel_projection(x)  # (batch_size, 128, audio_length // 16)
        
        # Generate audio through upsampling blocks
        audio = self.upsampling_blocks(x)  # (batch_size, 1, audio_length)
        
        return audio.squeeze(1)  # (batch_size, audio_length)

class Discriminator(nn.Module):
    def __init__(self, audio_length=AUDIO_LENGTH):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(1, 16, normalization=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv1d(512, 1, 4, stride=1, padding=0)
        )
    
    def forward(self, audio):
        # Input shape: (batch_size, audio_length)
        audio = audio.unsqueeze(1)  # Add channel dimension
        return self.model(audio) 