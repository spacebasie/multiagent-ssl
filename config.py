# config.py

"""
Configuration file for the VICReg model, training, and evaluation.
Supports both centralized and decentralized/federated simulations.
"""

# --- Device Configuration ---
DEVICE = "cuda"  # Use "cuda" if a GPU is available, otherwise "cpu"

# --- Simulation Parameters ---
# To run a centralized simulation, set NUM_AGENTS = 1.
# The total number of training epochs will be (COMMUNICATION_ROUNDS * LOCAL_EPOCHS).
NUM_AGENTS = 10               # Number of agents in the network
COMMUNICATION_ROUNDS = 50     # Number of training rounds
LOCAL_EPOCHS = 1              # Number of local epochs per agent per round
NON_IID_ALPHA = 0.5           # Dirichlet distribution alpha for non-IID split
                              # Smaller alpha = more non-IID. Not used if NUM_AGENTS = 1.

# --- Model Parameters ---
BACKBONE_MODEL = "resnet18"
PROJECTION_INPUT_DIM = 512
PROJECTION_HIDDEN_DIM = 2048
PROJECTION_OUTPUT_DIM = 2048

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
