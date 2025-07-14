# data.py

"""
Handles data loading and transformations specifically for the CIFAR-10 dataset.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from lightly.transforms.vicreg_transform import VICRegTransform
from torchvision.transforms import v2

def get_dataloaders(dataset_path: str, batch_size: int, num_workers: int, input_size: int):
    """
    Creates and returns the DataLoaders for CIFAR-10.
    """
    # Transform for self-supervised pre-training (with two views)
    transform_vicreg = VICRegTransform(input_size=input_size)
    pretrain_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path, download=True, train=True, transform=transform_vicreg
    )
    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )

    # Standard transforms for evaluation
    transform_eval = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    # Training and test sets for evaluation
    train_dataset_eval = torchvision.datasets.CIFAR10(
        root=dataset_path, download=True, train=True, transform=transform_eval
    )
    test_dataset_eval = torchvision.datasets.CIFAR10(
        root=dataset_path, download=True, train=False, transform=transform_eval
    )

    train_loader_eval = DataLoader(
        train_dataset_eval, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader_eval = DataLoader(
        test_dataset_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return pretrain_dataloader, train_loader_eval, test_loader_eval
