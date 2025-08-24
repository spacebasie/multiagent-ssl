# custom_datasets.py
import os
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch
import numpy as np

class CustomDataset(Dataset):
    """
    Generic dataset wrapper for datasets that return (x, y).
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            # The VICRegTransform returns two images, otherwise it's one
            img = self.transform(img)
        return img, label


class OfficeHomeDataset(Dataset):
    """
    Office-Home dataset loader.
    """
    DOMAINS = ["Art", "Clipart", "Product", "RealWorld"]

    def __init__(self, root_dir, transform=None, selected_domains=None, num_classes=None):
        self.root_dir = root_dir
        self.transform = transform

        if selected_domains is None:
            selected_domains = self.DOMAINS

        self.samples = []
        self.class_to_idx = {}

        # Discover all classes first
        all_classes = set()
        for domain in selected_domains:
            domain_path = os.path.join(root_dir, domain)
            if os.path.exists(domain_path):
                # Filter out non-directory files like .DS_Store
                subdirs = [d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))]
                all_classes.update(subdirs)

        sorted_classes = sorted(list(all_classes))
        if num_classes is not None and num_classes > 0:
            selected_classes = sorted_classes[:num_classes]
        else:
            selected_classes = sorted_classes

        self.classes = selected_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}

        for domain in selected_domains:
            domain_path = os.path.join(root_dir, domain)
            if not os.path.exists(domain_path):
                raise ValueError(f"Domain path {domain_path} not found in dataset root.")

            for class_name in selected_classes:
                class_path = os.path.join(domain_path, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        item = (img_path, self.class_to_idx[class_name])
                        self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # The transform is now applied by the wrapper CustomDataset
        return img, label


def get_officehome_train_test_loaders(root_dir, num_agents=1, batch_size=64, num_workers=4, train_transform=None,
                                      eval_transform=None, test_split=0.2, num_classes=None):
    """
    Loads Office-Home dataset, using different transforms for train and test splits.
    """
    # Create two separate dataset instances with different transforms
    full_train_dataset = OfficeHomeDataset(root_dir=root_dir, num_classes=num_classes)
    full_test_dataset = OfficeHomeDataset(root_dir=root_dir, num_classes=num_classes)

    # Split indices
    num_total_samples = len(full_train_dataset)
    indices = list(range(num_total_samples))
    split_idx = int(num_total_samples * (1 - test_split))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Create train and test subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    # Create agent-specific training dataloaders
    agent_train_indices = torch.randperm(len(train_dataset)).tolist()
    train_chunk_size = len(agent_train_indices) // num_agents
    agent_train_dataloaders = []
    for i in range(num_agents):
        start_idx = i * train_chunk_size
        end_idx = (i + 1) * train_chunk_size if i < num_agents - 1 else len(agent_train_indices)
        subset_indices = agent_train_indices[start_idx:end_idx]
        agent_train_subset = Subset(train_dataset, subset_indices)
        agent_train_dataloaders.append(
            DataLoader(CustomDataset(agent_train_subset, transform=train_transform), batch_size=batch_size,
                       shuffle=True, num_workers=num_workers)
        )

    # Create agent-specific test dataloaders
    agent_test_indices = torch.randperm(len(test_dataset)).tolist()
    test_chunk_size = len(agent_test_indices) // num_agents
    agent_test_dataloaders = []
    for i in range(num_agents):
        start_idx = i * test_chunk_size
        end_idx = (i + 1) * test_chunk_size if i < num_agents - 1 else len(agent_test_indices)
        subset_indices = agent_test_indices[start_idx:end_idx]
        agent_test_subset = Subset(test_dataset, subset_indices)
        agent_test_dataloaders.append(
            DataLoader(CustomDataset(agent_test_subset, transform=eval_transform), batch_size=batch_size, shuffle=False,
                       num_workers=num_workers)
        )

    # Create full evaluation loaders
    train_loader_eval = DataLoader(CustomDataset(train_dataset, transform=eval_transform), batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    test_loader_eval = DataLoader(CustomDataset(test_dataset, transform=eval_transform), batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

    return agent_train_dataloaders, agent_test_dataloaders, train_loader_eval, test_loader_eval


# Add this function for the DECENTRALIZED mode (personalized evaluation)
def get_officehome_domain_split_loaders_personalized(root_dir, num_agents, batch_size, num_workers, train_transform,
                                                     eval_transform, num_classes=None):
    """
    Creates dataloaders where each agent is assigned a specific domain from OfficeHome.
    Returns agent-specific training and testing dataloaders for personalized evaluation.
    """
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    agent_train_dataloaders = []
    agent_test_dataloaders = []

    for i in range(num_agents):
        agent_domain = domains[i % len(domains)]
        print(f"Assigning domain '{agent_domain}' to Agent {i}...")

        agent_dataset = OfficeHomeDataset(
            root_dir=root_dir,
            num_classes=num_classes,
            selected_domains=[agent_domain]
        )

        test_size = int(0.2 * len(agent_dataset))
        train_size = len(agent_dataset) - test_size
        agent_train_set, agent_test_set = random_split(agent_dataset, [train_size, test_size])

        agent_train_dataloaders.append(
            DataLoader(CustomDataset(agent_train_set, transform=train_transform), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers)
        )
        agent_test_dataloaders.append(
            DataLoader(CustomDataset(agent_test_set, transform=eval_transform), batch_size=batch_size, shuffle=False,
                       num_workers=num_workers)
        )

    return agent_train_dataloaders, agent_test_dataloaders


# Add this function for the FEDERATED mode (global evaluation)
def get_officehome_domain_split_loaders_global(root_dir, num_agents, batch_size, num_workers, train_transform,
                                               eval_transform, num_classes=None):
    """
    Creates dataloaders where each agent is assigned a specific domain from OfficeHome.
    Also returns global evaluation loaders containing data from all domains.
    """
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    agent_train_dataloaders = []
    all_train_sets = []
    all_test_sets = []

    for i in range(num_agents):
        agent_domain = domains[i % len(domains)]
        print(f"Assigning domain '{agent_domain}' to Agent {i}...")

        agent_dataset = OfficeHomeDataset(
            root_dir=root_dir,
            num_classes=num_classes,
            selected_domains=[agent_domain]
        )

        test_size = int(0.2 * len(agent_dataset))
        train_size = len(agent_dataset) - test_size
        agent_train_set, agent_test_set = random_split(agent_dataset, [train_size, test_size])

        all_train_sets.append(agent_train_set)
        all_test_sets.append(agent_test_set)

        agent_train_dataloaders.append(
            DataLoader(CustomDataset(agent_train_set, transform=train_transform), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers)
        )

    global_train_dataset = torch.utils.data.ConcatDataset(all_train_sets)
    global_test_dataset = torch.utils.data.ConcatDataset(all_test_sets)

    train_loader_eval = DataLoader(CustomDataset(global_train_dataset, transform=eval_transform), batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    test_loader_eval = DataLoader(CustomDataset(global_test_dataset, transform=eval_transform), batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

    return agent_train_dataloaders, train_loader_eval, test_loader_eval


def get_officehome_hierarchical_loaders(
        root_dir, num_neighborhoods, agents_per_neighborhood, batch_size,
        num_workers, train_transform, eval_transform, test_split=0.2
):
    """
    Creates dataloaders for a hierarchical heterogeneity experiment.
    - Inter-Neighborhood: Label Skew (specialized on classes)
    - Intra-Neighborhood: Domain Shift (specialized on domains)
    """
    print("Creating hierarchical dataloaders for Office-Home...")
    domains = OfficeHomeDataset.DOMAINS
    if agents_per_neighborhood > len(domains):
        raise ValueError(
            f"agents_per_neighborhood ({agents_per_neighborhood}) cannot exceed the number of available domains ({len(domains)}).")

    # 1. Get all classes and partition them among neighborhoods
    full_dataset = OfficeHomeDataset(root_dir=root_dir)
    all_classes = full_dataset.classes
    np.random.shuffle(all_classes)
    class_partitions = np.array_split(all_classes, num_neighborhoods)

    agent_train_dataloaders = []
    agent_test_dataloaders = []
    agent_id_counter = 0

    # 2. Iterate through each neighborhood to assign classes and domains
    for n_idx in range(num_neighborhoods):
        neighborhood_classes = class_partitions[n_idx]
        print(
            f"\nNeighborhood {n_idx}: Specializing in {len(neighborhood_classes)} classes -> {neighborhood_classes.tolist()}")

        # 3. Within a neighborhood, assign each agent a unique domain
        for a_idx in range(agents_per_neighborhood):
            agent_domain = domains[a_idx % len(domains)]
            print(f"  - Agent {agent_id_counter}: Assigned domain '{agent_domain}'")

            # Create a dataset for this specific agent (specific classes, specific domain)
            agent_dataset = OfficeHomeDataset(
                root_dir=root_dir,
                selected_domains=[agent_domain],
                num_classes=None  # We will filter by class name manually
            )

            # Manually filter the samples to only include the neighborhood's specialist classes
            class_indices_map = {cls: i for i, cls in enumerate(agent_dataset.classes)}
            target_class_indices = {class_indices_map[cls] for cls in neighborhood_classes if cls in class_indices_map}

            agent_samples = [s for s in agent_dataset.samples if s[1] in target_class_indices]
            agent_dataset.samples = agent_samples

            # Split into train and test sets for this specific agent
            test_size = int(test_split * len(agent_dataset))
            train_size = len(agent_dataset) - test_size

            if train_size > 0 and test_size > 0:
                agent_train_set, agent_test_set = random_split(agent_dataset, [train_size, test_size])

                agent_train_dataloaders.append(
                    DataLoader(CustomDataset(agent_train_set, transform=train_transform), batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
                )
                agent_test_dataloaders.append(
                    DataLoader(CustomDataset(agent_test_set, transform=eval_transform), batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
                )
            else:
                print(f"  - WARNING: Agent {agent_id_counter} has insufficient data and will be skipped.")

            agent_id_counter += 1

    print("\nHierarchical dataloaders created successfully.")
    return agent_train_dataloaders, agent_test_dataloaders