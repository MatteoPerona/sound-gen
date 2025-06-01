import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, n_mels, audio_length):
        super(Generator, self).__init__()
        
        # Input shape: (batch_size, n_mels=96, time_steps=1292)
        self.n_mels = n_mels
        self.time_steps = 1292
        
        # Initial dense layer to project mel spectrogram
        self.initial = nn.Sequential(
            nn.Linear(n_mels * self.time_steps, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024)
        )
        
        # Transposed convolution layers for upsampling
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 1, 1024)
            nn.ConvTranspose1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Final layer to match exact audio length
        self.final = nn.Linear(16384, audio_length)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure input is the right shape
        if len(x.shape) == 3:  # (batch_size, n_mels, time_steps)
            x = x.view(batch_size, -1)  # Flatten to (batch_size, n_mels * time_steps)
        elif len(x.shape) == 2:  # (batch_size, n_mels * time_steps)
            pass  # Already flattened
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
            
        # Project mel spectrogram
        x = self.initial(x)
        
        # Reshape for convolutions
        x = x.view(batch_size, 1, -1)
        x = self.conv_layers(x)
        
        # Reshape and match exact audio length
        x = x.view(batch_size, -1)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, audio_length):
        super(Discriminator, self).__init__()
        
        # Initial projection to reduce dimensionality
        self.initial = nn.Sequential(
            nn.Linear(audio_length, 16384),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 1, 16384)
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Final classification layer
        self.final = nn.Linear(1024, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure input is the right shape
        if len(x.shape) == 2:  # (batch_size, audio_length)
            pass  # Already flattened
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
            
        # Initial projection
        x = self.initial(x)
        
        # Reshape for convolutions
        x = x.view(batch_size, 1, -1)
        x = self.conv_layers(x)
        
        # Final classification
        x = x.view(batch_size, -1)
        x = self.final(x)
        return x
