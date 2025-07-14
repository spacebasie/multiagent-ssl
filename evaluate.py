# evaluate.py

"""
Contains functions for evaluating the pre-trained model, now with corrected data unpacking.
"""

import torch
from torch import nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def linear_evaluation(model, proj_output_dim, train_loader, test_loader, epochs, device):
    """Runs the linear evaluation protocol on the frozen backbone."""
    print("\n--- Starting Linear Evaluation ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    classifier = nn.Linear(proj_output_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
            # Correct unpacking for standard datasets
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                representations = model.forward_backbone(images)
            predictions = classifier(representations)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Correct unpacking for standard datasets
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
            # Correct unpacking for standard datasets
            images = images.to(device)
            features = model.forward_backbone(images)
            train_features.append(features)
            train_labels.append(labels)

    train_features = F.normalize(torch.cat(train_features, dim=0), dim=1)
    train_labels = torch.cat(train_labels, dim=0).to(device)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Correct unpacking for standard datasets
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

def log_tsne_plot(model, dataloader, device, epoch):
    """Generates and logs a t-SNE plot of the backbone's feature representations."""
    print("--- Generating t-SNE plot ---")
    model.eval()
    features_list, labels_list = [], []
    num_samples, samples_gathered = 1000, 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Correct unpacking for standard datasets
            images = images.to(device)
            features = model.forward_backbone(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            samples_gathered += len(images)
            if samples_gathered >= num_samples:
                break

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    tsne_features = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='tab10', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_title(f"t-SNE of Feature Embeddings at Epoch {epoch}")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    wandb.log({f"eval/tsne_plot": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)
    print("--- t-SNE plot logged to W&B ---")
