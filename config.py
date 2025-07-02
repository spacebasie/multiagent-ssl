# config.py

"""
Configuration file for the VICReg model, training, and evaluation.
Includes settings for both centralized and federated learning.
"""

# --- Training Mode ---
TRAINING_MODE = 'centralized' # Can be 'centralized' or 'federated'

# --- Device Configuration ---
DEVICE = "cuda"  # Use "cuda" if a GPU is available, otherwise "cpu"

# --- Federated Learning Parameters ---
# These are only used if TRAINING_MODE is 'federated'
NUM_AGENTS = 10               # Number of agents in the network
COMMUNICATION_ROUNDS = 50     # Number of federated training rounds
NON_IID_ALPHA = 0.5           # Dirichlet distribution alpha for non-IID split
                              # Smaller alpha = more non-IID

# --- Model Parameters ---
BACKBONE_MODEL = "resnet18"
PROJECTION_INPUT_DIM = 512
PROJECTION_HIDDEN_DIM = 2048
PROJECTION_OUTPUT_DIM = 2048

# --- Pre-training Parameters ---
# For 'centralized', this is the total number of epochs.
# For 'federated', this is the number of local epochs per round.
EPOCHS = 1 # Changed from LOCAL_EPOCHS for clarity
CENTRALIZED_EPOCHS = 200 # Total epochs for centralized training

# --- General Training Parameters ---
BATCH_SIZE = 256
NUM_WORKERS = 2

# --- VICReg Loss Hyperparameters ---
LAMBDA = 32.0
MU = 32.0
NU = 1.0

# --- Evaluation Parameters ---
EVAL_EPOCHS = 50
KNN_K = 200
KNN_TEMPERATURE = 0.1

# --- Dataset Configuration ---
DATASET_PATH = "datasets/cifar10"
INPUT_SIZE = 32

# --- File Paths ---
MODEL_SAVE_PATH = "vicreg_backbone.pth" # Filename will be adjusted by the script
