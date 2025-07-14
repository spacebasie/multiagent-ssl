# data.py

"""
Handles all data loading and transformations for different datasets.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from lightly.transforms.vicreg_transform import VICRegTransform
from lightly.data import LightlyDataset
from torchvision.transforms import v2 # Import v2 transforms

def get_dataloaders(dataset_name: str, dataset_path: str, batch_size: int, num_workers: int, input_size: int):
    """
    Creates and returns the DataLoaders for pre-training and evaluation for a given dataset.
    """
    transform_vicreg = VICRegTransform(input_size=input_size)

    if dataset_name == 'cifar10':
        normalize_transform = v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        pretrain_dataset = torchvision.datasets.CIFAR10(
            root=dataset_path, download=True, train=True, transform=transform_vicreg
        )
        train_dataset_eval = torchvision.datasets.CIFAR10(
            root=dataset_path, download=True, train=True
        )
        test_dataset_eval = torchvision.datasets.CIFAR10(
            root=dataset_path, download=True, train=False
        )

    elif dataset_name == 'imagenette':
        normalize_transform = v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        pretrain_dataset = LightlyDataset(
            input_dir=f"{dataset_path}/imagenette2-160/train",
            transform=transform_vicreg
        )
        train_dataset_eval = LightlyDataset(
            input_dir=f"{dataset_path}/imagenette2-160/train"
        )
        test_dataset_eval = LightlyDataset(
            input_dir=f"{dataset_path}/imagenette2-160/val"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Use the modern v2 transforms to avoid the deprecation warning
    transform_eval = v2.Compose([
        v2.Resize((input_size, input_size)),
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
        train_dataset_eval, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    test_loader_eval = DataLoader(
        test_dataset_eval, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    return pretrain_dataloader, train_loader_eval, test_loader_eval
