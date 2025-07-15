# data.py

"""
Handles data loading and transformations for CIFAR datasets.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from lightly.transforms.vicreg_transform import VICRegTransform
from torchvision.transforms import v2


def get_dataloaders(dataset_name: str, dataset_path: str, batch_size: int, num_workers: int, input_size: int):
    """
    Creates and returns the DataLoaders for CIFAR-10 or CIFAR-100.
    """
    transform_vicreg = VICRegTransform(input_size=input_size)

    if dataset_name == 'cifar10':
        normalize_transform = v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        DatasetClass = torchvision.datasets.CIFAR10

    elif dataset_name == 'cifar100':
        normalize_transform = v2.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )
        DatasetClass = torchvision.datasets.CIFAR100

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please use 'cifar10' or 'cifar100'.")

    # Create datasets
    pretrain_dataset = DatasetClass(
        root=dataset_path, download=True, train=True, transform=transform_vicreg
    )
    train_dataset_eval = DatasetClass(
        root=dataset_path, download=True, train=True
    )
    test_dataset_eval = DatasetClass(
        root=dataset_path, download=True, train=False
    )

    # Apply evaluation transform
    transform_eval = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize_transform,
    ])
    train_dataset_eval.transform = transform_eval
    test_dataset_eval.transform = transform_eval

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

    return pretrain_dataloader, train_loader_eval, test_loader_eval
