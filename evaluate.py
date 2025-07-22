# evaluate.py

"""
Contains functions for evaluating the pre-trained model on CIFAR-10.
"""

import torch
from torch import nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt


def linear_evaluation(model, proj_output_dim, train_loader, test_loader, epochs, device, agent_id=None):
    """Runs the linear evaluation protocol on the frozen backbone."""
    print("\n--- Starting Linear Evaluation ---")
    for param in model.backbone.parameters():
        param.requires_grad = False
    classifier = nn.Linear(proj_output_dim, 10).to(device) # CIFAR-10 has 10 classes or 100 for CIFAR-100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
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
