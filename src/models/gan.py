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

class MRFBlock(nn.Module):
# Lack of Multi-Scale Context
# Problem: Single receptive field convolutions may not capture both local (harmonic) 
# and global (formant) structure. This helps mitigate that.
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=5, padding=2, dilation=1),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
                nn.LeakyReLU(0.2),
            ),
        ])
    def forward(self, x):
        out = sum(conv(x) for conv in self.convs) / len(self.convs)
        return out + x


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

        # Mel projections for re-injection
        self.mel_proj1 = nn.Conv1d(n_mels, 128, kernel_size=1)
        self.mel_proj2 = nn.Conv1d(n_mels, 64, kernel_size=1)
        self.mel_proj3 = nn.Conv1d(n_mels, 32, kernel_size=1)
        self.mel_proj4 = nn.Conv1d(n_mels, 16, kernel_size=1)
        
        # Upsampling layers and corresponding MRFBlocks
        self.up1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.mrf1 = MRFBlock(128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.mrf2 = MRFBlock(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.mrf3 = MRFBlock(32)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.up4 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.mrf4 = MRFBlock(16)
        self.bn4 = nn.BatchNorm1d(16)
        
        self.final_conv = nn.Conv1d(16 + n_mels, 1, kernel_size=7, stride=1, padding=3)
        self.final_act = nn.Tanh()

    def forward(self, mel_spec):
        x = mel_spec.squeeze(1)  # (B, n_mels, T)
        mel_skip = x

        x = self.initial_conv(x)  # (B, 128, T)

        # Stage 1
        x = self.up1(x)
        x = self.bn1(x)
        mel1 = F.interpolate(mel_skip, size=x.shape[-1], mode='linear', align_corners=False)
        x = x + self.mel_proj1(mel1)
        x = self.mrf1(x)

        # Stage 2
        x = self.up2(x)
        x = self.bn2(x)
        mel2 = F.interpolate(mel_skip, size=x.shape[-1], mode='linear', align_corners=False)
        x = x + self.mel_proj2(mel2)
        x = self.mrf2(x)

        # Stage 3
        x = self.up3(x)
        x = self.bn3(x)
        mel3 = F.interpolate(mel_skip, size=x.shape[-1], mode='linear', align_corners=False)
        x = x + self.mel_proj3(mel3)
        x = self.mrf3(x)

        # Stage 4
        x = self.up4(x)
        x = self.bn4(x)
        mel4 = F.interpolate(mel_skip, size=x.shape[-1], mode='linear', align_corners=False)
        x = x + self.mel_proj4(mel4)
        x = self.mrf4(x)

        # Final skip connection (as before)
        if mel_skip.shape[-1] != x.shape[-1]:
            mel_skip = F.interpolate(mel_skip, size=x.shape[-1], mode='linear', align_corners=False)
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

    def spectral_convergence(self, real_mag, fake_mag):
        return torch.norm(real_mag - fake_mag, p='fro') / torch.norm(real_mag, p='fro')

    def stft_loss(self, x_real, x_fake):
        loss = 0.0
        for fft_size, hop_size, win_size in self.stft_sizes:
            window = torch.hann_window(win_size, device=x_real.device)

            real_stft = torch.stft(
                x_real, n_fft=fft_size, hop_length=hop_size,
                win_length=win_size, window=window, return_complex=True
            )
            fake_stft = torch.stft(
                x_fake, n_fft=fft_size, hop_length=hop_size,
                win_length=win_size, window=window, return_complex=True
            )

            real_mag = torch.abs(real_stft)
            fake_mag = torch.abs(fake_stft)

            sc_loss = self.spectral_convergence(real_mag, fake_mag)
            mag_loss = F.l1_loss(fake_mag, real_mag)

            loss += sc_loss + mag_loss

        return loss / len(self.stft_sizes)

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
            loss_fm * 5.0 +  # λ_fm
            loss_stft * 40.0 +  # λ_stft
            loss_mel * 150.0 +  # λ_mel
            loss_wav * 0.0     # λ_wav
        )
        
        return {
            'total': total_loss,
            'stft': loss_stft,
            'mel': loss_mel,
            'fm': loss_fm,
            'wav': loss_wav
        }

