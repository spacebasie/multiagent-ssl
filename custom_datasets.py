# custom_datasets.py
import os
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split


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
            img = self.transform(img)
        return img, label


class OfficeHomeDataset(Dataset):
    """
    Office-Home dataset loader.
    Folder structure:
        root/
            Art/
                Class1/
                Class2/
            Clipart/
            Product/
            RealWorld/
    This implementation can randomly split data for agents.
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
                all_classes.update(os.listdir(domain_path))

        # Sort classes for consistency and select a subset if requested
        sorted_classes = sorted(list(all_classes))
        if num_classes is not None and num_classes > 0:
            selected_classes = sorted_classes[:num_classes]
        else:
            selected_classes = sorted_classes

        # Create a mapping for the selected classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}

        # Gather samples only from selected domains and classes
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
        if self.transform:
            img = self.transform(img)
        return img, label


def get_officehome_train_test_loaders(root_dir, num_agents=1, batch_size=64, num_workers=4, transform=None,
                                      test_split=0.2, num_classes=None):
    """
    Loads Office-Home dataset, splits it into train/test, and then creates
    agent-specific dataloaders for both training and testing.
    """
    full_dataset = OfficeHomeDataset(root_dir=root_dir, transform=transform, num_classes=num_classes)

    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # --- Split training data for each agent ---
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    train_chunk_size = len(train_indices) // num_agents
    agent_train_subsets = []
    for i in range(num_agents):
        start_idx = i * train_chunk_size
        end_idx = (i + 1) * train_chunk_size if i < num_agents - 1 else len(train_indices)
        subset_indices = train_indices[start_idx:end_idx]
        agent_train_subsets.append(Subset(train_dataset, subset_indices))

    agent_train_dataloaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for subset in agent_train_subsets
    ]

    # --- Split testing data for each agent ---
    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)
    test_chunk_size = len(test_indices) // num_agents
    agent_test_subsets = []
    for i in range(num_agents):
        start_idx = i * test_chunk_size
        end_idx = (i + 1) * test_chunk_size if i < num_agents - 1 else len(test_indices)
        subset_indices = test_indices[start_idx:end_idx]
        agent_test_subsets.append(Subset(test_dataset, subset_indices))

    agent_test_dataloaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for subset in agent_test_subsets
    ]

    train_loader_eval = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_eval = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return agent_train_dataloaders, agent_test_dataloaders, train_loader_eval, test_loader_eval