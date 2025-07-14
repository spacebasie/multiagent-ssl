# data_splitter.py

"""
Handles the partitioning of a dataset into non-IID subsets for federated learning.
Now compatible with both torchvision datasets and LightlyDataset.
"""

import torch
import numpy as np
from torch.utils.data import Subset, Dataset
from torchvision.datasets import CIFAR10


def _get_dataset_labels(dataset: Dataset) -> np.ndarray:
    """
    Gets the labels from a dataset, handling different dataset types.
    """
    if hasattr(dataset, 'targets'):
        # For torchvision datasets like CIFAR10
        return np.array(dataset.targets)
    elif hasattr(dataset, 'get_labels'):
        # For datasets with a dedicated get_labels method
        return np.array(dataset.get_labels())
    else:
        # For other datasets like LightlyDataset, we iterate
        print("Dataset has no .targets attribute, iterating to get labels...")
        labels = []
        # Note: This can be slow for very large datasets.
        for _, label, _ in dataset:
            labels.append(label)
        return np.array(labels)


def split_data(dataset: Dataset, num_agents: int, non_iid_alpha: float) -> list[Subset]:
    """
    Splits a dataset into non-IID partitions for a specified number of agents.

    Args:
        dataset (Dataset): The full dataset to be partitioned.
        num_agents (int): The number of agents to distribute the data among.
        non_iid_alpha (float): The concentration parameter for the Dirichlet
                               distribution, controlling the non-IID level.

    Returns:
        list[Subset]: A list where each element is a PyTorch Subset representing
                      the local data for one agent.
    """
    print(f"Splitting data into {num_agents} non-IID partitions with alpha={non_iid_alpha}...")

    # Get labels using our robust helper function
    labels = _get_dataset_labels(dataset)
    num_classes = len(np.unique(labels))

    # Create a list of indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Use a Dirichlet distribution to determine the proportion of each class for each agent
    class_distribution = np.random.dirichlet([non_iid_alpha] * num_classes, num_agents)

    agent_indices = [[] for _ in range(num_agents)]

    # Distribute the indices of each class according to the Dirichlet proportions
    for c in range(num_classes):
        proportions = class_distribution[:, c]
        num_samples_per_agent = (proportions * len(class_indices[c])).astype(int)

        remainder = len(class_indices[c]) - num_samples_per_agent.sum()
        if remainder > 0:
            for i in range(num_agents - 1, -1, -1):
                if num_samples_per_agent[i] > 0:
                    num_samples_per_agent[i] += remainder
                    break

        np.random.shuffle(class_indices[c])

        start_idx = 0
        for i in range(num_agents):
            end_idx = start_idx + num_samples_per_agent[i]
            agent_indices[i].extend(class_indices[c][start_idx:end_idx])
            start_idx = end_idx

    # Create a PyTorch Subset for each agent using their assigned indices
    agent_datasets = [Subset(dataset, indices) for indices in agent_indices]

    print("Data splitting complete.")
    return agent_datasets
