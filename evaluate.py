# evaluate.py

"""
Contains functions for evaluating the pre-trained model on CIFAR-10.
"""

import torch
from torch import nn
import torch.nn.functional as F
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def linear_evaluation(model, proj_output_dim, train_loader, test_loader, epochs, device, agent_id=None):
    """Runs the linear evaluation protocol on the frozen backbone."""
    print("\n--- Starting Linear Evaluation ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    classifier = nn.Linear(proj_output_dim, 10).to(device) # CIFAR-10 has 10 classes or 100 for CIFAR-100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
            if isinstance(images, list):
                images = images[0]
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                representations = model.forward_backbone(images)
            predictions = classifier(representations)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    correct, total = 0, 0
    classifier.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            if isinstance(images, list):
                images = images[0]
            images, labels = images.to(device), labels.to(device)
            representations = model.forward_backbone(images)
            predictions = classifier(representations)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Final Linear Evaluation Test Accuracy: {accuracy:.2f}%")
    return accuracy

def knn_evaluation(model, train_loader, test_loader, device, k=20, temperature=0.1):
    """Runs the k-Nearest Neighbors (kNN) evaluation on the frozen backbone."""
    print("\n--- Starting kNN Evaluation ---")
    model.eval()
    train_features, train_labels = [], []
    with torch.no_grad():
        for images, labels in train_loader:
            if isinstance(images, list):
                images = images[0]
            images = images.to(device)
            features = model.forward_backbone(images)
            train_features.append(features)
            train_labels.append(labels)

    train_features = F.normalize(torch.cat(train_features, dim=0), dim=1)
    train_labels = torch.cat(train_labels, dim=0).to(device)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test_features = F.normalize(model.forward_backbone(images), dim=1)
            similarity_matrix = torch.matmul(test_features, train_features.T) / temperature
            _, indices = similarity_matrix.topk(k, dim=1, largest=True, sorted=True)
            k_neighbor_labels = train_labels[indices]
            predictions = torch.mode(k_neighbor_labels, dim=1).values
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Final kNN Test Accuracy: {accuracy:.2f}%")
    return accuracy


def plot_tsne(model, test_loader, device, plot_title="t-SNE Visualization", save_html_path=None):
    """
    Generates and logs a t-SNE plot to W&B and optionally saves an interactive HTML version.
    """
    print(f"\n--- Generating and logging '{plot_title}' to W&B ---")
    model.eval()
    all_features = []
    all_labels = []

    # 1. Get all features and labels from the test set
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features = model.forward_backbone(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    if len(all_features) > 2000:
        print("Dataset is large, using a random subset of 2000 points for t-SNE.")
        indices = np.random.choice(len(all_features), 2000, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1), max_iter=1000)
    tsne_results = tsne.fit_transform(all_features)
    print("t-SNE finished.")

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis', alpha=0.7)

    class_names = None
    dataset = test_loader.dataset
    while hasattr(dataset, 'dataset'):
        if hasattr(dataset, 'classes') and dataset.classes:
            class_names = dataset.classes
            break
        dataset = dataset.dataset
    if hasattr(dataset, 'classes') and dataset.classes:
        class_names = dataset.classes

    if class_names:
        legend_elements = scatter.legend_elements()
        num_legend_entries = len(legend_elements[0])
        ax.legend(legend_elements[0], class_names[:num_legend_entries], title="Classes")
    else:
        ax.legend(*scatter.legend_elements(), title="Classes")

    ax.set_title(plot_title)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True)

    # --- START: New additions ---
    # 1. Fix the layout to prevent the legend from being cut off
    fig.tight_layout()

    # 2. Optionally save an interactive HTML version of the plot
    if save_html_path:
        try:
            from wandb import util
            import plotly.io as pio

            print(f"Converting plot to interactive HTML...")
            plotly_fig = util.matplotlib_to_plotly(fig)
            pio.write_html(plotly_fig, file=save_html_path)
            print(f"Interactive plot saved to {save_html_path}")
        except ImportError:
            print(
                "Warning: Could not save interactive plot. Please ensure 'plotly' is installed (`pip install plotly`).")
    # --- END: New additions ---

    wandb.log({plot_title: fig})
    print(f"'{plot_title}' logged to W&B.")
    plt.close(fig)

def plot_pca(model, test_loader, device, plot_title="PCA Visualization"):
    """
    Generates and logs a PCA plot of the model's representations to W&B.
    """
    print(f"\n--- Generating and logging '{plot_title}' to W&B ---")
    model.eval()
    all_features = []
    all_labels = []

    # 1. Get all features and labels from the test set
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features = model.forward_backbone(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # 2. Apply PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(all_features)
    print("PCA finished.")

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], c=all_labels, cmap='viridis', alpha=0.7)

    # Logic to find and display class names in the legend
    class_names = None
    dataset = test_loader.dataset
    while hasattr(dataset, 'dataset'):
        if hasattr(dataset, 'classes') and dataset.classes:
            class_names = dataset.classes
            break
        dataset = dataset.dataset
    if hasattr(dataset, 'classes') and dataset.classes:
        class_names = dataset.classes

    if class_names:
        legend_elements = scatter.legend_elements()
        num_legend_entries = len(legend_elements[0])
        ax.legend(legend_elements[0], class_names[:num_legend_entries], title="Classes")
    else:
        ax.legend(*scatter.legend_elements(), title="Classes")

    ax.set_title(plot_title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    fig.tight_layout()

    # 4. Log the plot to Weights & Biases
    wandb.log({plot_title: fig})
    print(f"'{plot_title}' logged to W&B.")
    plt.close(fig)