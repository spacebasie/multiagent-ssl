# data_splitter.py

"""
Handles the partitioning of a dataset into non-IID subsets for federated learning.
Now compatible with both torchvision datasets and LightlyDataset.
"""

import torch
import numpy as np
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import copy


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


def get_domain_shift_dataloaders(
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        num_agents: int,
        agent_transforms: list,
        dataset_config: dict

) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Creates dataloaders for the domain shift experiment by applying different transforms.
    """
    print(f"Creating {num_agents} domain-shifted dataloaders...")
    if len(agent_transforms) != num_agents:
        raise ValueError(
            f"The number of agent_transforms ({len(agent_transforms)}) must match num_agents ({num_agents}).")

    train_len = len(train_dataset)
    train_indices = torch.randperm(train_len).tolist()
    train_split_size = train_len // num_agents

    test_len = len(test_dataset)
    test_indices = torch.randperm(test_len).tolist()
    test_split_size = test_len // num_agents

    train_dataloaders = []
    test_dataloaders = []

    # Define a standard evaluation transform (ToTensor + Normalize)
    # This will be applied AFTER the domain-specific corruption.
    # Note: Using CIFAR-10 stats. This could be made more general if needed.
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=dataset_config['normalize_mean'], std=dataset_config['normalize_std'])
    ])

    for i in range(num_agents):
        # --- Create Training Dataloader for Agent i ---
        start_train = i * train_split_size
        end_train = (i + 1) * train_split_size if i < num_agents - 1 else train_len
        agent_train_indices = train_indices[start_train:end_train]

        agent_train_dataset = copy.deepcopy(train_dataset)
        agent_train_dataset.transform = agent_transforms[i]
        agent_train_subset = Subset(agent_train_dataset, agent_train_indices)
        train_dataloaders.append(
            DataLoader(agent_train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        )

        # --- Create Test Dataloader for Agent i ---
        start_test = i * test_split_size
        end_test = (i + 1) * test_split_size if i < num_agents - 1 else test_len
        agent_test_indices = test_indices[start_test:end_test]

        agent_test_dataset = copy.deepcopy(test_dataset)

        # --- THE FIX IS HERE ---
        # 1. Get the agent's specific domain shift (e.g., GaussianBlur).
        #    We assume the transform is a PreTransform object.
        if hasattr(agent_transforms[i], 'pre_transform'):
            domain_shift_transform = agent_transforms[i].pre_transform
        else:  # Fallback if not a PreTransform object
            domain_shift_transform = T.Compose([])  # No-op

        # 2. Create a new transform pipeline for the test set that combines
        #    the domain shift AND the standard evaluation transform.
        final_test_transform = T.Compose([
            domain_shift_transform,  # Apply blur/rotation first (PIL -> PIL)
            eval_transform  # Then convert to tensor and normalize (PIL -> Tensor)
        ])
        agent_test_dataset.transform = final_test_transform
        # --- END OF FIX ---

        agent_test_subset = Subset(agent_test_dataset, agent_test_indices)
        test_dataloaders.append(
            DataLoader(agent_test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )

    print("Domain-shifted dataloaders created successfully.")
    return train_dataloaders, test_dataloaders


# FILE: data_splitter.py

def _partition_dataset_by_distribution(dataset, num_agents, class_distribution):
    """
    Helper function to partition a dataset based on a given Dirichlet distribution.
    """
    labels = _get_dataset_labels(dataset)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    agent_indices = [[] for _ in range(num_agents)]

    for c in range(num_classes):
        proportions = class_distribution[:, c]
        num_class_samples = len(class_indices[c])
        num_samples_per_agent = (proportions * num_class_samples).astype(int)

        remainder = num_class_samples - num_samples_per_agent.sum()
        if remainder > 0:
            for i in np.argsort(num_samples_per_agent)[::-1]:
                if remainder == 0:
                    break
                num_samples_per_agent[i] += 1
                remainder -= 1

        np.random.shuffle(class_indices[c])

        start_idx = 0
        for i in range(num_agents):
            end_idx = start_idx + num_samples_per_agent[i]
            agent_indices[i].extend(class_indices[c][start_idx:end_idx])
            start_idx = end_idx

    return [Subset(dataset, indices) for indices in agent_indices]


def split_train_test_data_personalized(train_dataset, test_dataset, num_agents, non_iid_alpha):
    """
    Splits both training and testing datasets into non-IID partitions for personalized evaluation.
    It uses the same class distribution for both splits for each agent.
    """
    print(
        f"Splitting train and test data into {num_agents} personalized non-IID partitions with alpha={non_iid_alpha}...")

    labels = _get_dataset_labels(train_dataset)
    num_classes = len(np.unique(labels))
    class_distribution = np.random.dirichlet([non_iid_alpha] * num_classes, num_agents)

    agent_train_datasets = _partition_dataset_by_distribution(train_dataset, num_agents, class_distribution)
    agent_test_datasets = _partition_dataset_by_distribution(test_dataset, num_agents, class_distribution)

    print("Personalized train and test data splitting complete.")
    return agent_train_datasets, agent_test_datasets