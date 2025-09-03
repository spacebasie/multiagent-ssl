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
import numpy as np
from training import get_consensus_model
from network import gossip_average
from evaluate import linear_evaluation, plot_tsne, calculate_representation_angles, plot_angle_evolution, knn_evaluation  # We can reuse this for the final eval
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
        eval_every,
        alignment_strength,
        alignment_only=False,
        classifier_only=False
):
    """
    The main training loop for the alignment-regularized protocol.
    """
    num_agents = len(agent_backbones)

    backbone_optimizers = [torch.optim.SGD(backbone.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4) for
                           backbone in agent_backbones]
    classifier_optimizers = [torch.optim.Adam(classifier.parameters(), lr=0.001) for classifier in agent_classifiers]

    public_data_iter = iter(public_dataloader)
    angle_history = []

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

        if not classifier_only:
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
                alignment_loss = alignment_strength * F.mse_loss(Z_public_i, avg_public_embeddings[i].detach())
                alignment_loss.backward()
                backbone_optimizers[i].step()

        if not alignment_only:
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

            angles_this_round = calculate_representation_angles(
                agent_backbones=agent_backbones,
                public_dataloader=public_dataloader,
                global_test_loader=global_test_loader,
                device=device
            )
            if angles_this_round:
                angle_history.append(angles_this_round)
                median = np.median(angles_this_round)
                q1 = np.percentile(angles_this_round, 25)
                q3 = np.percentile(angles_this_round, 75)

                # Log the statistics as individual metrics
                wandb.log({
                    "eval/angle_median": median,
                    "eval/angle_25th_percentile": q1,
                    "eval/angle_75th_percentile": q3
                        }, step=round_num + 1)
            # --- NEW: After all rounds are complete, generate the final evolution plot ---
        plot_angle_evolution(angle_history, eval_every)

    return agent_backbones, agent_classifiers


def final_combo_evaluation(
        final_backbones,
        final_classifiers,
        global_train_loader,
        global_test_loader,
        device
):
    """
    Performs the final, rigorous evaluation using a single, shared classifier.
    This is called only once at the end of all training rounds.
    """
    print("\n--- Starting Final Rigorous Evaluation with Shared Classifier ---")
    num_agents = len(final_backbones)

    # 1. Create the single, shared classifier by averaging all final classifiers.
    #    - For the 'disconnected' run, this averages misaligned classifiers.
    #    - For the 'connected' run, this averages already-aligned classifiers.
    print("Creating final shared consensus classifier...")
    shared_classifier = get_consensus_model(final_classifiers, device)

    # 2. Evaluate each agent's personalized backbone with this single shared classifier.
    final_accuracies = []
    for i in range(num_agents):
        print(f"Evaluating Agent {i}'s personalized backbone with the shared classifier...")
        # The evaluate_combo_model helper is perfect for this task
        acc = evaluate_combo_model(final_backbones[i], shared_classifier, global_test_loader, device)
        final_accuracies.append(acc)
        wandb.summary[f"agent_{i}_final_combo_accuracy"] = acc

    # 3. Log the final average accuracy.
    avg_acc = sum(final_accuracies) / len(final_accuracies) if final_accuracies else 0
    wandb.summary["average_final_combo_accuracy"] = avg_acc
    print(f"\nFinal Average Collaborative Accuracy: {avg_acc:.2f}%")

    print("\n--- Evaluating with k-NN ---")
    final_knn_accuracies = []
    for i in range(num_agents):
        print(f"Running k-NN evaluation for Agent {i}'s backbone...")
        knn_acc = knn_evaluation(
            model=final_backbones[i],
            train_loader=global_train_loader,  # Used to build the feature bank
            test_loader=global_test_loader,
            device=device
        )
        final_knn_accuracies.append(knn_acc)
        wandb.summary[f"agent_{i}_final_knn_accuracy"] = knn_acc

    avg_knn_acc = sum(final_knn_accuracies) / len(final_knn_accuracies) if final_knn_accuracies else 0
    wandb.summary["average_final_knn_accuracy"] = avg_knn_acc
    print(f"\nFinal Average k-NN Accuracy: {avg_knn_acc:.2f}%")


    print("\n--- Generating Final t-SNE Visualizations ---")
    for i, backbone in enumerate(final_backbones):
        csv_filename = f"tsne_data_agent_{i}.csv"
        plot_tsne(
            model=backbone,
            test_loader=global_test_loader,
            device=device,
            plot_title=f"Final t-SNE for Agent {i}'s Backbone",
            save_csv_path=csv_filename
        )