# config.py

"""
Unified configuration file for both benchmark and federated runs.
"""

# --- Device Configuration ---
DEVICE = "cuda"

# --- Dataset Configuration ---
DATASET_NAME = "cifar10" # Change to 'cifar10' to switch
DATASET_PATH = "datasets"
INPUT_SIZE = 128

# --- Centralized Benchmark Run Parameters ---
BENCHMARK_EPOCHS = 800
BATCH_SIZE = 256
NUM_WORKERS = 8
LEARNING_RATE = 0.05

# --- Federated Learning Parameters ---
# These are used when --num_agents > 1
NUM_AGENTS = 10
COMMUNICATION_ROUNDS = 100
LOCAL_EPOCHS = 2
NON_IID_ALPHA = 0.5

# --- Model Parameters ---
BACKBONE_MODEL = "resnet18"
PROJECTION_INPUT_DIM = 512
PROJECTION_HIDDEN_DIM = 2048
PROJECTION_OUTPUT_DIM = 2048

# --- VICReg Loss Hyperparameters ---
LAMBDA = 25.0
MU = 25.0
NU = 1.0

# --- Evaluation Parameters ---
EVAL_EPOCHS = 50
KNN_K = 200
KNN_TEMPERATURE = 0.1

# --- File Paths ---
MODEL_SAVE_PATH = "vicreg_backbone.pth"
