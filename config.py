# config.py

"""
Configuration file for the VICReg model, training, and evaluation.

This file centralizes all hyperparameters and settings to allow for easy
adjustments and to ensure consistency across different parts of the project.
"""

# --- Device Configuration ---
DEVICE = "cuda"  # Use "cuda" if a GPU is available, otherwise "cpu"

# --- Model Parameters ---
# Configuration for the ResNet-18 backbone and VICReg projection head
BACKBONE_MODEL = "resnet18"
PROJECTION_INPUT_DIM = 512  # Output dimension of ResNet-18 backbone
PROJECTION_HIDDEN_DIM = 2048 # Hidden dimension of the projection head
PROJECTION_OUTPUT_DIM = 2048 # Final output dimension of the projection head

# --- Pre-training Parameters ---
# Settings for the self-supervised pre-training phase
PRETRAIN_EPOCHS = 200 # Number of epochs for self-supervised learning
BATCH_SIZE = 256      # Batch size for pre-training and evaluation
NUM_WORKERS = 2       # Number of workers for the DataLoader

# --- VICReg Loss Hyperparameters ---
# Coefficients for the different components of the VICReg loss function
# lambda and mu control the weight of the invariance and variance terms
LAMBDA = 32.0
MU = 32.0
NU = 1.0  # nu controls the covariance term, typically fixed at 1.0

# --- Evaluation Parameters ---
# Settings for the linear and kNN evaluation phases
EVAL_EPOCHS = 50       # Number of epochs for training the linear classifier
KNN_K = 200            # Number of nearest neighbors for kNN evaluation
KNN_TEMPERATURE = 0.1  # Temperature scaling for kNN similarity matrix

# --- Dataset Configuration ---
# Details for the dataset used in pre-training and evaluation
DATASET_PATH = "datasets/cifar10" # Directory to store the CIFAR-10 dataset
INPUT_SIZE = 32 # Input image size for CIFAR-10

# --- File Paths ---
# Path for saving the trained model weights
MODEL_SAVE_PATH = "vicreg_backbone.pth"