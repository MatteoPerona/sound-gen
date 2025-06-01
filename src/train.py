import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

from models.gan import Generator, Discriminator
from utils import log_audio_samples, save_checkpoint
from data.dataset import MelSpectrogramDataset
from config import *

def train():
    # Initialize wandb
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dataset and dataloader
    dataset = MelSpectrogramDataset(mels_dir="data/mels", audio_dir="data/audio")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = Generator(n_mels=N_MELS, audio_length=AUDIO_LENGTH).to(device)
    discriminator = Discriminator(audio_length=AUDIO_LENGTH).to(device)
    
    # Loss function - using BCEWithLogitsLoss for better numerical stability
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        for i, (mel_specs, real_audio) in enumerate(dataloader):
            batch_size = mel_specs.shape[0]
            mel_specs = mel_specs.to(device)
            real_audio = real_audio.to(device)
            
            # Ground truths - no need for sigmoid as BCEWithLogitsLoss includes it
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate audio from mel spectrograms
            gen_audio = generator(mel_specs)
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_audio), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_audio), valid)
            fake_loss = adversarial_loss(discriminator(gen_audio.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Log progress
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch,
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
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