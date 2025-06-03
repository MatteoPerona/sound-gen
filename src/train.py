import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

from models.gan import Generator, Discriminator, VocoderLoss
from utils import log_audio_samples, save_checkpoint
from data.dataset import MelSpectrogramDataset
from config import *

import random

def train():
    # Initialize wandb
    print("initializing wandb...")
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    
    # Set device
    print("setting device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"============== GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("============== No GPU available, using CPU.")
    
    # Initialize dataset and dataloader
    print("initializing dataset and dataloader...")
    dataset = MelSpectrogramDataset(mels_dir="data/clean/mels", audio_dir="data/clean/audio")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    print("initializing models...")
    generator = Generator(n_mels=N_MELS, audio_length=AUDIO_LENGTH).to(device)
    discriminator = Discriminator(audio_length=AUDIO_LENGTH).to(device)
    
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    vocoder_loss = VocoderLoss().to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=G_LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=D_LEARNING_RATE, betas=(BETA1, BETA2))
    
    # Training loop
    print("starting training loop...")
    for epoch in range(NUM_EPOCHS):
        # Wrap dataloader with tqdm for progress bar
        for i, (mel_specs, real_audio) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            batch_size = mel_specs.shape[0]
            mel_specs = mel_specs.to(device)
            real_audio = real_audio.to(device)
            
            # Ground truths with label smoothing for real samples
            real_labels = torch.full((batch_size, 1), 0.9).to(device)  # Label smoothing for real samples
            fake_labels = torch.zeros(batch_size, 1).to(device)  # Keep fake labels at 0
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate audio from mel spectrograms
            gen_audio = generator(mel_specs)
            
            # Get discriminator predictions and features for both real and fake
            pred_fake, fake_features = discriminator(gen_audio, return_features=True)
            pred_real, real_features = discriminator(real_audio, return_features=True)
            
            # Calculate all losses
            losses = vocoder_loss(real_audio, gen_audio, real_features, fake_features)
            
            # Add adversarial loss
            pred_fake = pred_fake.mean(dim=2)  # (batch, 1)
            # g_loss = adversarial_loss(pred_fake, real_labels) + losses['total']
            g_loss = losses['total']
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            d_real_labels = real_labels.clone()
            d_fake_labels = fake_labels.clone()
            
            # LABEL FLIPPING
            flip_mask = torch.rand(batch_size, 1) < 0.1
            d_real_labels[flip_mask] = 0.0  # Flip real labels to fake
            d_fake_labels[flip_mask] = 0.9  # Flip fake labels to real
            
            # Get discriminator predictions (no need for features here)
            pred_real = discriminator(real_audio, return_features=False).mean(dim=2)  # (batch, 1)
            pred_fake = discriminator(gen_audio.detach(), return_features=False).mean(dim=2)  # (batch, 1)
            real_loss = adversarial_loss(pred_real, d_real_labels)  # D should predict 0.9 for real samples
            fake_loss = adversarial_loss(pred_fake, d_fake_labels)  # D should predict 0 for fake samples

            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Log progress
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                    f"[STFT loss: {losses['stft'].item():.4f}] [Mel loss: {losses['mel'].item():.4f}] "
                    f"[FM loss: {losses['fm'].item():.4f}] [Wav loss: {losses['wav'].item():.4f}]"
                )
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch,
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
                    "stft_loss": losses['stft'].item(),
                    "mel_loss": losses['mel'].item(),
                    "fm_loss": losses['fm'].item(),
                    "wav_loss": losses['wav'].item(),
                })
                
                # Log sample audio
                if i % 500 == 0:
                    log_audio_samples(gen_audio[0], real_audio[0], epoch * len(dataloader) + i)
        
        # Save checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                generator,
                optimizer_G,
                epoch,
                g_loss.item(),
                f"checkpoints/generator_epoch_{epoch}.pt"
            )
            save_checkpoint(
                discriminator,
                optimizer_D,
                epoch,
                d_loss.item(),
                f"checkpoints/discriminator_epoch_{epoch}.pt"
            )

if __name__ == "__main__":
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Start training
    train()