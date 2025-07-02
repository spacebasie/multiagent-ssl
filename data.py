# data.py

"""
Handles data loading and transformations for the VICReg project.

This module provides functions to get the necessary DataLoaders for both
pre-training (with custom augmentations) and evaluation (with standard transforms).
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from lightly.transforms.vicreg_transform import VICRegTransform

def get_dataloaders(batch_size: int, num_workers: int, input_size: int, path: str) -> tuple:
    """
    Creates and returns the DataLoaders for pre-training and evaluation.

    Args:
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses to use for data loading.
        input_size (int): The size of the input images (e.g., 32 for CIFAR-10).
        path (str): The root directory for the dataset.

    Returns:
        tuple: A tuple containing:
               - pretrain_dataloader (DataLoader): For self-supervised pre-training.
               - train_loader_eval (DataLoader): For training the linear classifier.
               - test_loader_eval (DataLoader): For testing the classifier and kNN.
    """
    # 1. Transform for self-supervised pre-training (VICReg specific augmentations)
    transform_vicreg = VICRegTransform(input_size=input_size)
    pretrain_dataset = torchvision.datasets.CIFAR10(
        root=path, download=True, transform=transform_vicreg
    )
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    # 2. Standard transforms for linear and kNN evaluation
    transform_eval = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Training set for evaluation
    train_dataset_eval = torchvision.datasets.CIFAR10(
        root=path, download=True, train=True, transform=transform_eval
    )
    train_loader_eval = DataLoader(
        train_dataset_eval,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Test set for evaluation
    test_dataset_eval = torchvision.datasets.CIFAR10(
        root=path, download=True, train=False, transform=transform_eval
    )
    test_loader_eval = DataLoader(
        test_dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return pretrain_dataloader, train_loader_eval, test_loader_eval