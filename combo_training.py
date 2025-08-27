# combo_training.py
"""
Contains the training and evaluation loop for the novel alignment-regularized
collaborative classification protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb
from network import gossip_average
from evaluate import linear_evaluation  # We can reuse this for the final eval
from config import KNN_K, KNN_TEMPERATURE


def gossip_average_tensors(tensor_list: list, adj_matrix: dict):
    """
    Performs one round of gossip averaging on a list of tensors.
    This is a production-ready way to average the public embeddings.
    """
    num_agents = len(tensor_list)
    if num_agents == 0:
        return []

    old_tensors = [t.clone().detach() for t in tensor_list]
    new_tensors = []

    for i in range(num_agents):
        neighbor_indices = [j for j, connected in enumerate(adj_matrix[i]) if connected]

        # Start with its own tensor
        summed_tensor = old_tensors[i].clone()

        # Add tensors from neighbors
        for neighbor_idx in neighbor_indices:
            if neighbor_idx != i:
                summed_tensor += old_tensors[neighbor_idx]

        # Average by the number of connections (self + neighbors)
        avg_tensor = summed_tensor / len(neighbor_indices)
        new_tensors.append(avg_tensor)

    return new_tensors


def gossip_average_classifier(agent_classifiers: list, adj_matrix: dict):
    """
    Performs one round of gossip averaging on a list of linear classifiers.
    This function reuses the logic of averaging state_dicts.
    """
    num_agents = len(agent_classifiers)
    if num_agents == 0:
        return []

    old_states = [copy.deepcopy(model.state_dict()) for model in agent_classifiers]

    for i in range(num_agents):
        neighbor_indices = [j for j, connected in enumerate(adj_matrix[i]) if connected]
        new_state_dict = copy.deepcopy(old_states[i])

        for neighbor_idx in neighbor_indices:
            if neighbor_idx != i:
                for key in new_state_dict.keys():
                    new_state_dict[key] += old_states[neighbor_idx][key]

        num_connections = len(neighbor_indices)
        for key in new_state_dict.keys():
            new_state_dict[key] = torch.div(new_state_dict[key], num_connections)

        agent_classifiers[i].load_state_dict(new_state_dict)

    return agent_classifiers


def evaluate_combo_model(backbone, classifier, test_loader, device):
    """
    Helper function to evaluate a single backbone + classifier pair on a test set.
    """
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if isinstance(images, list):  # Handle VICRegTransform output
                images = images[0]
            images, labels = images.to(device), labels.to(device)

            features = backbone.forward_backbone(images)
            predictions = classifier(features)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def alignment_collaborative_training(
        agent_backbones,
        agent_classifiers,
        agent_train_dataloaders,
        global_test_loader,
        public_dataloader,
        adj_matrix,
        vicreg_criterion,
        classifier_criterion,
        device,
        comm_rounds,
        local_epochs,
        learning_rate,
        eval_every
):
    """
    The main training loop for the alignment-regularized protocol.
    """
    num_agents = len(agent_backbones)

    backbone_optimizers = [torch.optim.SGD(backbone.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4) for
                           backbone in agent_backbones]
    classifier_optimizers = [torch.optim.Adam(classifier.parameters(), lr=0.001) for classifier in agent_classifiers]

    public_data_iter = iter(public_dataloader)

    for round_num in range(comm_rounds):
        print(f"\n--- Round {round_num + 1}/{comm_rounds} ---")

        # --- Step 1: Local Unsupervised & Supervised Updates ---
        for i in range(num_agents):
            print(f"Training Agent {i} locally...")
            agent_backbones[i].train()
            agent_classifiers[i].train()

            for _ in range(local_epochs):
                for x_views, y_private in agent_train_dataloaders[i]:
                    x0, x1 = x_views[0].to(device), x_views[1].to(device)
                    y_private = y_private.to(device)

                    # 1a. Backbone Update (Unsupervised VICReg)
                    backbone_optimizers[i].zero_grad()
                    z0 = agent_backbones[i](x0)
                    z1 = agent_backbones[i](x1)
                    vicreg_loss = vicreg_criterion(z0, z1)['loss']
                    vicreg_loss.backward()
                    backbone_optimizers[i].step()

                    # 1b. Classifier Update (Supervised)
                    classifier_optimizers[i].zero_grad()
                    with torch.no_grad():
                        features = agent_backbones[i].forward_backbone(x0)
                    predictions = agent_classifiers[i](features)
                    classifier_loss = classifier_criterion(predictions, y_private)
                    classifier_loss.backward()
                    classifier_optimizers[i].step()

        # --- Step 2 & 3: Alignment using Public Data ---
        print("Performing alignment step...")
        try:
            x_public_views, _ = next(public_data_iter)
        except StopIteration:
            public_data_iter = iter(public_dataloader)
            x_public_views, _ = next(public_data_iter)

        x_public = x_public_views[0].to(device)

        with torch.no_grad():
            public_embeddings = [bb.forward_backbone(x_public) for bb in agent_backbones]

        avg_public_embeddings = gossip_average_tensors(public_embeddings, adj_matrix)

        for i in range(num_agents):
            backbone_optimizers[i].zero_grad()
            Z_public_i = agent_backbones[i].forward_backbone(x_public)
            alignment_loss = F.mse_loss(Z_public_i, avg_public_embeddings[i].detach())
            alignment_loss.backward()
            backbone_optimizers[i].step()

        # --- Step 4: Communication Phase 2 - Collaborate on Classifier ---
        print("Averaging classifiers...")
        agent_classifiers = gossip_average_classifier(agent_classifiers, adj_matrix)

        # --- Periodic Evaluation ---
        if (round_num + 1) % eval_every == 0:
            print(f"\n--- Evaluating at Round {round_num + 1} ---")
            total_acc = 0
            for i in range(num_agents):
                acc = evaluate_combo_model(agent_backbones[i], agent_classifiers[i], global_test_loader, device)
                wandb.log({f"eval/agent_{i}_combo_accuracy": acc}, step=round_num + 1)
                total_acc += acc
            avg_acc = total_acc / num_agents
            wandb.log({"eval/avg_combo_accuracy": avg_acc}, step=round_num + 1)
            print(f"Average Collaborative Accuracy: {avg_acc:.2f}%")

    return agent_backbones, agent_classifiers