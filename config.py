# config.py

"""
Configuration file for the VICReg model, training, and evaluation.
"""

# --- Device Configuration ---
DEVICE = "cuda"

# --- Dataset Configuration ---
# Change this to 'cifar10' to switch back
DATASET_NAME = "imagenette"
DATASET_PATH = "datasets" # Root folder for all datasets
INPUT_SIZE = 128          # Using 128px input size as per the benchmark

# --- Benchmark Run Parameters ---
BENCHMARK_EPOCHS = 800
BATCH_SIZE = 256
NUM_WORKERS = 8
LEARNING_RATE = 0.05

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
KNN_K = 20                # Using k=20 as per the benchmark
KNN_TEMPERATURE = 0.1

# --- File Paths ---
MODEL_SAVE_PATH = "vicreg_backbone.pth"
