import wandb

# Wandb configuration
WANDB_PROJECT = "mel-spectrogram-gan"
WANDB_ENTITY = None  # Set this to your wandb username

# Data parameters
SAMPLE_RATE = 22050
AUDIO_LENGTH = 22050 * 30  # 30 seconds of audio
N_MELS = 80
# HOP_LENGTH = 256
HOP_LENGTH = 512
WIN_LENGTH = 1024
F_MIN = 0
F_MAX = 8000

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Model parameters
LATENT_DIM = 100
GEN_CHANNELS = 64
DISC_CHANNELS = 64

# Wandb logging
def init_wandb():
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "sample_rate": SAMPLE_RATE,
            "audio_length": AUDIO_LENGTH,
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "beta1": BETA1,
            "beta2": BETA2,
            "latent_dim": LATENT_DIM,
            "gen_channels": GEN_CHANNELS,
            "disc_channels": DISC_CHANNELS,
        }
    ) 