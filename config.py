# config.py

"""
Configuration file for the VICReg model, training, and evaluation.
This version is simplified and dedicated to CIFAR-10.
"""

# --- Device Configuration ---
DEVICE = "cuda"

# --- Dataset Configuration ---
DATASET_PATH = "datasets/cifar10"
INPUT_SIZE = 32

# --- Federated Learning Parameters ---
NUM_AGENTS = 10
COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 2
NON_IID_ALPHA = 0.5

# --- Centralized Training Parameters (when NUM_AGENTS = 1) ---
EPOCHS = 200 # A reasonable number for a single run

# --- General Training Parameters ---
BATCH_SIZE = 256
NUM_WORKERS = 4
LEARNING_RATE = 0.01 # For centralized runs try 0.05

# --- Model Parameters ---
PROJECTION_INPUT_DIM = 512
PROJECTION_HIDDEN_DIM = 2048
PROJECTION_OUTPUT_DIM = 2048

# --- VICReg Loss Hyperparameters ---
LAMBDA = 25 # Your fine-tuned value
MU = 25   # Your fine-tuned value
NU = 1.0

# --- Evaluation Parameters ---
EVAL_EPOCHS = 50
KNN_K = 200
KNN_TEMPERATURE = 0.1
EVAL_EVERY = 5
