import wandb

# Wandb configuration
WANDB_PROJECT = "mel-spectrogram-gan"
WANDB_ENTITY = None  # Set this to your wandb username

# Data parameters
SAMPLE_RATE = 12000
AUDIO_IN_SECONDS = 15
AUDIO_LENGTH = SAMPLE_RATE * AUDIO_IN_SECONDS  
N_MELS = 96
HOP_LENGTH = 256
WIN_LENGTH = 512
F_MIN = 0
F_MAX = 6000

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
G_LEARNING_RATE = 1e-4
D_LEARNING_RATE = 5e-5
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
            "g_learning_rate": G_LEARNING_RATE,
            "d_learning_rate": D_LEARNING_RATE,
            "beta1": BETA1,
            "beta2": BETA2,
            "latent_dim": LATENT_DIM,
            "gen_channels": GEN_CHANNELS,
            "disc_channels": DISC_CHANNELS,
        }
    ) 