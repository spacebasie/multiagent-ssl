# combo_training.py

"""
Contains the training and evaluation loop for the alignment-regularized
collaborative classification protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb
import numpy as np
from training import agent_update  # We can reuse the unsupervised part
from network import gossip_average
from evaluate import knn_evaluation


def gossip_average_tensors(tensor_list: list, adj_matrix: dict):
    """
    Performs one round of gossip averaging on a list of tensors.
    This is a production-ready way to average the public embeddings.
    """
    num_agents = len(tensor_list)
    old_tensors = [t.clone() for t in tensor_list]
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


def alignment_collaborative_training(
        agent_backbones,
        agent_classifiers,
        agent_train_dataloaders,
        agent_test_dataloaders,
        public_dataloader,
        adj_matrix,
        vicreg_criterion,
        classifier_criterion,
        device,
        comm_rounds,
        local_epochs,
        learning_rate
):
    """
    The main training loop for the alignment-regularized protocol.
    """
    num_agents = len(agent_backbones)

    # Create optimizers for each backbone and classifier
    backbone_optimizers = [torch.optim.SGD(backbone.parameters(), lr=learning_rate, momentum=0.9) for backbone in
                           agent_backbones]
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
                for (x_private, x_private_aug), y_private in agent_train_dataloaders[i]:
                    x_private, x_private_aug, y_private = x_private.to(device), x_private_aug.to(device), y_private.to(
                        device)

                    # 1a. Backbone Update (Unsupervised)
                    backbone_optimizers[i].zero_grad()
                    z0 = agent_backbones[i](x_private)
                    z1 = agent_backbones[i](x_private_aug)
                    vicreg_loss = vicreg_criterion(z0, z1)['loss']
                    vicreg_loss.backward()
                    backbone_optimizers[i].step()

                    # 1b. Classifier Update (Supervised)
                    classifier_optimizers[i].zero_grad()
                    with torch.no_grad():
                        features = agent_backbones[i].forward_backbone(x_private)
                    predictions = agent_classifiers[i](features)
                    classifier_loss = classifier_criterion(predictions, y_private)
                    classifier_loss.backward()
                    classifier_optimizers[i].step()

        # --- Step 2 & 3: Alignment using Public Data ---
        print("Performing alignment step...")
        try:
            x_public, _ = next(public_data_iter)
        except StopIteration:
            public_data_iter = iter(public_dataloader)
            x_public, _ = next(public_data_iter)
        x_public = x_public.to(device)

        # Generate embeddings for the public batch
        with torch.no_grad():
            public_embeddings = [bb.forward_backbone(x_public) for bb in agent_backbones]

        # Communication Phase 1: Average the public embeddings
        # Note: This is a conceptual simplification. In a real implementation,
        # you would need a gossip protocol for tensors, not just model states.
        # For now, we simulate it with a simple average.
        Z_public_avg = torch.mean(torch.stack(public_embeddings), dim=0)

        # Perform corrective update
        for i in range(num_agents):
            backbone_optimizers[i].zero_grad()
            # We need to re-compute the embeddings to build the graph for backward()
            Z_public_i = agent_backbones[i].forward_backbone(x_public)
            alignment_loss = F.mse_loss(Z_public_i, Z_public_avg.detach())
            alignment_loss.backward()
            backbone_optimizers[i].step()

        # --- Step 4: Communication Phase 2 - Collaborate on Classifier ---
        print("Averaging classifiers...")
        agent_classifiers = gossip_average_classifier(agent_classifiers, adj_matrix)

        # --- Evaluation (simplified for this draft) ---
        # A full implementation would add periodic evaluation here.

    # At the end, all agents have their personalized backbones, but they share one
    # collaboratively trained classifier (we can just take agent 0's).
    final_classifier = agent_classifiers[0]
    return agent_backbones, final_classifier