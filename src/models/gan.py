import torch
import torch.nn as nn
import torch.nn.functional as F
from config import AUDIO_LENGTH, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX

# How to mitigate this?
# Residual/Skip Connections:
# Use skip connections from early layers (or even from the input mel) to later layers or the output. This helps preserve original frequency information.
# DONE!

# Careful Kernel/Stride Choices:
# Use kernel sizes and strides that preserve resolution and context.

# Multi-Receptive Field Blocks:
# Use parallel convolutions with different dilations (as in HiFi-GAN) to capture both local and global context.

# Feature-wise Conditioning:
# Consider re-injecting the mel spectrogram (or features from it) at multiple points in the generator.

# Monitor Feature Maps:
# Visualize intermediate feature maps to ensure frequency information is preserved.

class Snake(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                Snake(),
                nn.Conv1d(channels, channels, kernel_size, padding=d, dilation=d),
                Snake(),
                nn.Conv1d(channels, channels, kernel_size, padding=1, dilation=3)
            )
            for d in dilations
        ])
    
    def forward(self, x):
        out = sum([conv(x) for conv in self.convs]) / len(self.convs)
        # Ensure out and x have the same time dimension
        # i THINK this shouldnt matter, since we mainly just want to preserve x
        if out.shape[-1] != x.shape[-1]:
            min_len = min(out.shape[-1], x.shape[-1])
            out = out[..., :min_len]
            x = x[..., :min_len]
        return out + x

class Generator(nn.Module):
    def __init__(self, n_mels=N_MELS, audio_length=AUDIO_LENGTH):
        super(Generator, self).__init__()
        
        self.initial_conv = nn.Conv1d(n_mels, 128, kernel_size=7, padding=3)
        
        # Upsampling layers and corresponding ResBlocks
        self.up1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.rb1 = ResBlock(128, kernel_size=3, dilations=[1, 3, 5])
        self.bn1 = nn.BatchNorm1d(128)
        
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.rb2 = ResBlock(64, kernel_size=3, dilations=[1, 3, 5])
        self.bn2 = nn.BatchNorm1d(64)
        
        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.rb3 = ResBlock(32, kernel_size=3, dilations=[1, 3, 5])
        self.bn3 = nn.BatchNorm1d(32)
        
        self.up4 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.rb4 = ResBlock(16, kernel_size=3, dilations=[1, 3, 5])
        self.bn4 = nn.BatchNorm1d(16)
        
        self.final_conv = nn.Conv1d(16 + n_mels, 1, kernel_size=7, stride=1, padding=3)
        self.final_act = nn.Tanh()

    def forward(self, mel_spec):
        x = mel_spec.squeeze(1)  # (B, n_mels, T)
        mel_skip = x
        x = self.initial_conv(x)  # (B, 128, T)
        
        x = self.up1(x)
        x = self.bn1(x)
        x = self.rb1(x)
        
        x = self.up2(x)
        x = self.bn2(x)
        x = self.rb2(x)
        
        x = self.up3(x)
        x = self.bn3(x)
        x = self.rb3(x)
        
        x = self.up4(x)
        x = self.bn4(x)
        x = self.rb4(x)

        # Upsample mel_skip to match x's time dimension if needed
        if mel_skip.shape[-1] != x.shape[-1]:
            mel_skip = F.interpolate(mel_skip, size=x.shape[-1], mode='linear', align_corners=False)
        # Concatenate skip connection
        x = torch.cat([x, mel_skip], dim=1)
        
        x = self.final_conv(x)
        x = self.final_act(x)
        
        if x.shape[-1] != AUDIO_LENGTH:
            x = F.interpolate(x, size=AUDIO_LENGTH, mode='linear', align_corners=False)
        return x.squeeze(1)

class Discriminator(nn.Module):
    def __init__(self, audio_length=AUDIO_LENGTH):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.blocks = nn.ModuleList([
            nn.Sequential(*discriminator_block(1, 16, normalization=False)),
            nn.Sequential(*discriminator_block(16, 32)),
            nn.Sequential(*discriminator_block(32, 64)),
            nn.Sequential(*discriminator_block(64, 128)),
            nn.Sequential(*discriminator_block(128, 256)),
            nn.Sequential(*discriminator_block(256, 512)),
        ])
        
        self.final_conv = nn.Conv1d(512, 1, 4, stride=1, padding=0)
    
    def forward(self, audio, return_features=False):
        # Input shape: (batch_size, audio_length)
        audio = audio.unsqueeze(1)  # Add channel dimension
        
        features = []
        x = audio
        
        # Pass through each block and collect features
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        if return_features:
            return x, features
            # return self.model(audio), features
        return x
        # return self.model(audio)

class VocoderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_sizes = [(1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)]
        
    def stft_loss(self, x_real, x_fake):
        loss = 0
        for fft_size, hop_size, win_size in self.stft_sizes:
            window = torch.hann_window(win_size, device=x_real.device)
            real_mag = torch.abs(torch.stft(
                x_real, fft_size, hop_length=hop_size, 
                win_length=win_size, window=window, return_complex=True
            ))
            fake_mag = torch.abs(torch.stft(
                x_fake, fft_size, hop_length=hop_size, 
                win_length=win_size, window=window, return_complex=True
            ))
            loss += F.l1_loss(real_mag, fake_mag)
        return loss

    def mel_loss(self, x_real, x_fake):
        window = torch.hann_window(WIN_LENGTH, device=x_real.device)
        mel_real = torch.stft(
            x_real, WIN_LENGTH, hop_length=HOP_LENGTH, 
            win_length=WIN_LENGTH, window=window, return_complex=True
        )
        mel_fake = torch.stft(
            x_fake, WIN_LENGTH, hop_length=HOP_LENGTH, 
            win_length=WIN_LENGTH, window=window, return_complex=True
        )
        mel_real = torch.abs(mel_real)
        mel_fake = torch.abs(mel_fake)
        return F.l1_loss(mel_real, mel_fake)
    
    def feature_matching_loss(self, real_features, fake_features):
        loss = 0
        for f_real, f_fake in zip(real_features, fake_features):
            loss += F.l1_loss(f_real, f_fake)
        return loss
    
    def forward(self, x_real, x_fake, real_features, fake_features):
        # Calculate all losses
        loss_stft = self.stft_loss(x_real, x_fake)
        loss_mel = self.mel_loss(x_real, x_fake)
        loss_fm = self.feature_matching_loss(real_features, fake_features)
        loss_wav = F.l1_loss(x_fake, x_real)  # Optional waveform loss
        
        # Combine losses with weights
        total_loss = (
            loss_fm * 10.0 +  # λ_fm
            loss_stft * 20.0 +  # λ_stft
            loss_mel * 75.0 +  # λ_mel
            loss_wav * 2.0     # λ_wav
        )
        
        return {
            'total': total_loss,
            'stft': loss_stft,
            'mel': loss_mel,
            'fm': loss_fm,
            'wav': loss_wav
        }

