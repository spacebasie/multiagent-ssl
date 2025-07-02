# data_splitter.py

"""
Handles the partitioning of a dataset into non-IID subsets for federated learning.

This module uses a Dirichlet distribution to simulate a realistic non-IID data
distribution among a network of agents. The `non_iid_alpha` parameter controls
the level of heterogeneity:
- A small alpha (e.g., 0.1) creates a highly skewed, non-IID distribution where
  each agent may only have data from a few classes.
- A large alpha (e.g., 100) creates a more uniform distribution that approaches IID.
"""

import torch
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


def split_data(dataset: CIFAR10, num_agents: int, non_iid_alpha: float) -> list[Subset]:
    """
    Splits a dataset into non-IID partitions for a specified number of agents.

    Args:
        dataset (CIFAR10): The full dataset to be partitioned.
        num_agents (int): The number of agents to distribute the data among.
        non_iid_alpha (float): The concentration parameter for the Dirichlet
                               distribution, controlling the non-IID level.

    Returns:
        list[Subset]: A list where each element is a PyTorch Subset representing
                      the local data for one agent.
    """
    print(f"Splitting data into {num_agents} non-IID partitions with alpha={non_iid_alpha}...")

    # Get the labels for the entire dataset
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    # Create a list of indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Use a Dirichlet distribution to determine the proportion of each class for each agent
    # This is the core of the non-IID split
    class_distribution = np.random.dirichlet([non_iid_alpha] * num_classes, num_agents)

    agent_indices = [[] for _ in range(num_agents)]

    # Distribute the indices of each class according to the Dirichlet proportions
    for c in range(num_classes):
        # Calculate the number of samples of class `c` for each agent
        proportions = class_distribution[:, c]
        num_samples_per_agent = (proportions * len(class_indices[c])).astype(int)

        # Ensure the total number of samples matches the class size
        # Add any remainder to the last agent that has a non-zero proportion
        remainder = len(class_indices[c]) - num_samples_per_agent.sum()
        if remainder > 0:
            for i in range(num_agents - 1, -1, -1):
                if num_samples_per_agent[i] > 0:
                    num_samples_per_agent[i] += remainder
                    break

        # Shuffle the indices of the current class to ensure random distribution
        np.random.shuffle(class_indices[c])

        # Assign indices to agents
        start_idx = 0
        for i in range(num_agents):
            end_idx = start_idx + num_samples_per_agent[i]
            agent_indices[i].extend(class_indices[c][start_idx:end_idx])
            start_idx = end_idx

    # Create a PyTorch Subset for each agent using their assigned indices
    agent_datasets = [Subset(dataset, indices) for indices in agent_indices]

    print("Data splitting complete.")
    return agent_datasets
