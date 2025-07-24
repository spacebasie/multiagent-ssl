# data.py

"""
Handles data loading and transformations for CIFAR datasets.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from lightly.transforms.vicreg_transform import VICRegTransform
import torchvision.transforms as T

# --- Dataset Registry ---
# To add a new dataset, add its details here.
DATASET_CONFIG = {
    'cifar10': {
        'class': torchvision.datasets.CIFAR10,
        'input_size': 32,
        'normalize_mean': [0.4914, 0.4822, 0.4465],
        'normalize_std': [0.2023, 0.1994, 0.2010],
    },
    'cifar100': {
        'class': torchvision.datasets.CIFAR100,
        'input_size': 32,
        'normalize_mean': [0.5071, 0.4867, 0.4408],
        'normalize_std': [0.2675, 0.2565, 0.2761],
    },
    'fashion_mnist': {
        'class': torchvision.datasets.FashionMNIST,
        'input_size': 28,
        'normalize_mean': [0.5], # Grayscale
        'normalize_std': [0.5],  # Grayscale
    },
    # Add other datasets here, e.g., SVHN
    # 'svhn': { ... }
}


def get_dataloaders(dataset_name: str, dataset_path: str, batch_size: int, num_workers: int):
    """
    Creates and returns DataLoaders for the specified dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets are: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset_name]
    DatasetClass = config['class']
    input_size = config['input_size']

    # The VICRegTransform for self-supervised pre-training
    transform_vicreg = VICRegTransform(input_size=input_size)

    # Create the training dataset for the VICReg model
    pretrain_dataset = DatasetClass(
        root=dataset_path, download=True, train=True, transform=transform_vicreg
    )

    # --- Create datasets for linear evaluation ---
    # The evaluation transform should NOT have augmentations like VICReg.
    # It should only convert to tensor and normalize.
    transform_eval = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=config['normalize_mean'], std=config['normalize_std'])
    ])

    # Create the train and test sets for evaluation
    train_dataset_eval = DatasetClass(
        root=dataset_path, download=True, train=True, transform=transform_eval
    )
    test_dataset_eval = DatasetClass(
        root=dataset_path, download=True, train=False, transform=transform_eval
    )

    # Create DataLoaders
    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    train_loader_eval = DataLoader(
        train_dataset_eval, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader_eval = DataLoader(
        test_dataset_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return pretrain_dataloader, train_loader_eval, test_loader_eval, config

