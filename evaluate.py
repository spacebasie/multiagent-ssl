# evaluate.py

"""
Contains functions for evaluating the pre-trained VICReg model.

This module provides two standard evaluation protocols for self-supervised models:
- linear_evaluation: Trains a linear classifier on top of the frozen backbone.
- knn_evaluation: Performs k-Nearest Neighbors classification on the representations.
"""

import torch
from torch import nn
import torch.nn.functional as F

def linear_evaluation(model, proj_output_dim, train_loader, test_loader, epochs, device):
    """
    Runs the linear evaluation protocol on the frozen backbone.

    A linear classifier is trained on the representations from the frozen backbone
    to assess the quality and linear separability of the learned features.

    Args:
        model (nn.Module): The pre-trained VICReg model.
        proj_output_dim (int): The output dimension of the backbone.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        epochs (int): Number of epochs to train the linear classifier.
        device (str): The device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        float: The final test accuracy of the linear classifier.
    """
    print("\n--- Starting Linear Evaluation ---")

    # Freeze the backbone to prevent its weights from being updated
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Create a simple linear classifier
    classifier = nn.Linear(proj_output_dim, 10).to(device) # CIFAR-10 has 10 classes

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Train the linear classifier
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad(): # Ensure backbone features are not updated
                representations = model.forward_backbone(images)

            predictions = classifier(representations)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Classifier Training Epoch: {epoch:02}, Loss: {avg_loss:.5f}")

    # Evaluate the trained classifier on the test set
    print("\nEvaluating on Test Set...")
    classifier.eval()
    model.eval()
    correct = 0
    total = 0
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

def knn_evaluation(model, train_loader, test_loader, device, k=200, temperature=0.1):
    """
    Runs the k-Nearest Neighbors (kNN) evaluation on the frozen backbone.

    This method assesses feature quality by measuring how well a kNN classifier
    performs on the representations from the frozen backbone.

    Args:
        model (nn.Module): The pre-trained VICReg model.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): The device to run the evaluation on ('cuda' or 'cpu').
        k (int): The number of nearest neighbors to consider.
        temperature (float): Temperature for scaling the similarity matrix.

    Returns:
        float: The final kNN test accuracy.
    """
    print("\n--- Starting kNN Evaluation ---")

    # Freeze the model backbone
    model.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Step 1: Extract features and labels from the entire training set
    print("Gathering training features for kNN...")
    train_features = []
    train_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            features = model.forward_backbone(images)
            train_features.append(features)
            train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0).to(device)

    # Normalize features for cosine similarity
    train_features = F.normalize(train_features, dim=1)

    # Step 2: Evaluate on the test set using the extracted training features
    print("Evaluating on test set using kNN...")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test_features = model.forward_backbone(images)
            test_features = F.normalize(test_features, dim=1)

            # Compute similarity between test and training features
            similarity_matrix = torch.matmul(test_features, train_features.T) / temperature

            # Get the top-k most similar training examples
            _, indices = similarity_matrix.topk(k, dim=1, largest=True, sorted=True)

            # Get the labels of these top-k neighbors
            k_neighbor_labels = train_labels[indices]

            # Predict the class via a majority vote
            predictions = torch.mode(k_neighbor_labels, dim=1).values
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Final kNN Test Accuracy: {accuracy:.2f}%")
    return accuracy

def log_tsne_plot(model, dataloader, device, epoch):
    """
    Generates and logs a t-SNE plot of the backbone's feature representations.
    """
    print("--- Generating t-SNE plot ---")
    model.eval()
    features_list = []
    labels_list = []
    # Use a subset of data to speed up t-SNE generation
    num_samples = 1000
    samples_gathered = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Adapt for LightlyDataset format: ((view1, view2), labels, fnames)
            # For t-SNE, we only need one view and the labels.
            (images, _), labels, _ = images, labels[0], labels[1]
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

    # Log the plot to Weights & Biases
    wandb.log({f"eval/tsne_plot": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)
    print("--- t-SNE plot logged to W&B ---")

