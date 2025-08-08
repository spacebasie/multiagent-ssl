import os
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset

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

    def __init__(self, root_dir, transform=None, selected_domains=None):
        self.root_dir = root_dir
        self.transform = transform

        # Use all domains if none specified
        if selected_domains is None:
            selected_domains = self.DOMAINS

        # Gather all samples from selected domains
        self.samples = []
        self.class_to_idx = None

        for domain in selected_domains:
            domain_path = os.path.join(root_dir, domain)
            if not os.path.exists(domain_path):
                raise ValueError(f"Domain path {domain_path} not found in dataset root.")
            domain_dataset = datasets.ImageFolder(domain_path)
            if self.class_to_idx is None:
                self.class_to_idx = domain_dataset.class_to_idx
            self.samples.extend(domain_dataset.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_officehome_loaders(root_dir, num_agents=1, batch_size=64, num_workers=4, transform=None):
    """
    Loads Office-Home dataset and splits it randomly across agents.
    Each agent gets an equal-sized random subset.
    """
    full_dataset = OfficeHomeDataset(root_dir=root_dir, transform=transform)

    # Shuffle indices
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    # Split into equal chunks
    chunk_size = len(indices) // num_agents
    agent_subsets = []
    for i in range(num_agents):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_agents - 1 else len(indices)
        subset_indices = indices[start_idx:end_idx]
        agent_subsets.append(Subset(full_dataset, subset_indices))

    # Create loaders
    agent_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for subset in agent_subsets
    ]
    return agent_loaders
