# decentralized_training.py

"""
Contains the training and evaluation loop for decentralized personalized learning,
specifically for the domain adaptation (Path 2) experiment.
"""

from evaluate import linear_evaluation, knn_evaluation, plot_tsne, plot_pca
from training import agent_update, get_consensus_model
from network import gossip_average
import wandb
from config import KNN_TEMPERATURE, KNN_K
from torch.utils.data import ConcatDataset, DataLoader


# In decentralized_training.py

def decentralized_personalized_training(
        agent_models,
        agent_train_dataloaders,
        agent_test_dataloaders,  # List of domain-specific test loaders
        adj_matrix,
        criterion,
        device,
        comm_rounds,
        local_epochs,
        eval_every,
        proj_input_dim,
        eval_epochs,
        learning_rate,
        global_train_loader_eval,  # This is the global training set for kNN feature bank
        global_test_loader_eval  # This is the global test set you want to evaluate on
):
    """
    Decentralized training with per-agent domain adaptation and BOTH personalized
    and global evaluation.
    """
    num_agents = len(agent_models)
    for round_num in range(comm_rounds):
        print(f"\n--- Round {round_num + 1}/{comm_rounds} ---")
        round_agg_loss = {}
        # Local training for each agent
        for i in range(num_agents):
            if len(agent_train_dataloaders[i]) == 0:
                print(f"Skipping Agent {i}: No data available for training.")
                continue
            if len(agent_train_dataloaders[i].dataset) > 0:
                print(f"Training Agent {i}...")
                loss_dict = agent_update(agent_models[i], agent_train_dataloaders[i], local_epochs, criterion, device,
                                         learning_rate)
                if loss_dict:
                    for k, v in loss_dict.items():
                        round_agg_loss[k] = round_agg_loss.get(k, 0.0) + v

        if round_agg_loss:
            num_successful_agents = num_agents
            for k in round_agg_loss:
                round_agg_loss[k] /= num_successful_agents
            wandb.log({f"train/avg_{k}": v for k, v in round_agg_loss.items()}, step=round_num + 1)

        print("Performing gossip averaging...")
        agent_models = gossip_average(agent_models, adj_matrix)

        if (round_num + 1) % eval_every == 0:
            print(f"\n--- Evaluating Models at Round {round_num + 1} ---")

            # --- 1. Personalized Evaluation (Existing Logic) ---
            total_personalized_acc = 0
            for i in range(num_agents):
                acc = knn_evaluation(
                    model=agent_models[i],
                    train_loader=agent_train_dataloaders[i],  # Uses agent's own skewed data for feature bank
                    test_loader=agent_test_dataloaders[i],  # Evaluates on agent's own skewed test set
                    device=device,
                    k=KNN_K,
                    temperature=KNN_TEMPERATURE
                )
                total_personalized_acc += acc
                wandb.log({f"eval/agent_{i}_personalized_knn_accuracy": acc}, step=round_num + 1)

            avg_personalized_acc = total_personalized_acc / num_agents
            print(f"Average Personalized KNN Test Accuracy: {avg_personalized_acc:.2f}%")
            wandb.log({"eval/avg_personalized_accuracy": avg_personalized_acc}, step=round_num + 1)

            # --- 2. Global Evaluation (New Logic) ---
            # Now, evaluate each specialized agent on the full, balanced global test set.
            total_global_acc = 0
            for i in range(num_agents):
                global_acc = knn_evaluation(
                    model=agent_models[i],
                    train_loader=global_train_loader_eval,  # Uses the full IID training set for the feature bank
                    test_loader=global_test_loader_eval,  # Evaluates on the full IID test set
                    device=device,
                    k=KNN_K,
                    temperature=KNN_TEMPERATURE
                )
                total_global_acc += global_acc
                wandb.log({f"eval/agent_{i}_global_knn_accuracy": global_acc}, step=round_num + 1)

            avg_global_acc = total_global_acc / num_agents
            print(f"Average Global KNN Test Accuracy: {avg_global_acc:.2f}%")
            wandb.log({"eval/avg_global_accuracy": avg_global_acc}, step=round_num + 1)

    # --- Final Evaluation ---
    print("\n--- Final Personalized and Global Evaluation ---")
    final_personalized_results = {}
    final_global_results = {}

    for i in range(num_agents):
        # Final Personalized Accuracy
        p_acc = knn_evaluation(
            model=agent_models[i],
            train_loader=agent_train_dataloaders[i],
            test_loader=agent_test_dataloaders[i],
            device=device, k=KNN_K, temperature=KNN_TEMPERATURE
        )
        final_personalized_results[f"agent_{i}_final_personalized_knn_acc"] = p_acc
        wandb.summary[f"agent_{i}_final_personalized_knn_accuracy"] = p_acc

        # Final Global Accuracy
        g_acc = knn_evaluation(
            model=agent_models[i],
            train_loader=global_train_loader_eval,
            test_loader=global_test_loader_eval,
            device=device, k=KNN_K, temperature=KNN_TEMPERATURE
        )
        final_global_results[f"agent_{i}_final_global_knn_acc"] = g_acc
        wandb.summary[f"agent_{i}_final_global_knn_accuracy"] = g_acc

    avg_final_personalized_acc = sum(final_personalized_results.values()) / len(final_personalized_results)
    avg_final_global_acc = sum(final_global_results.values()) / len(final_global_results)

    wandb.summary["average_final_personalized_accuracy"] = avg_final_personalized_acc
    wandb.summary["average_final_global_accuracy"] = avg_final_global_acc

    print(f"\nAverage Final Personalized Accuracy: {avg_final_personalized_acc:.2f}%")
    print(f"Average Final Global Accuracy: {avg_final_global_acc:.2f}%")


def evaluate_neighborhood_consensus(
        all_agent_models,
        all_agent_test_dataloaders,
        num_neighborhoods,
        agents_per_neighborhood,
        device,
        proj_input_dim,
        eval_epochs,
        batch_size  # Add batch_size as an argument
):
    """
    Performs a neighborhood-level consensus evaluation for the hierarchical experiment.
    """
    print("\n--- Starting Neighborhood-Level Consensus Evaluation ---")

    for n_idx in range(num_neighborhoods):
        print(f"\n--- Evaluating Neighborhood {n_idx} ---")

        # 1. Isolate the models and test loaders for the current neighborhood
        start_idx = n_idx * agents_per_neighborhood
        end_idx = start_idx + agents_per_neighborhood

        neighborhood_models = all_agent_models[start_idx:end_idx]
        neighborhood_test_loaders = all_agent_test_dataloaders[start_idx:end_idx]

        if not neighborhood_models:
            print(f"No active models in Neighborhood {n_idx}, skipping evaluation.")
            continue

        # 2. Create the consensus model for this neighborhood using the existing function
        print(f"Creating consensus model for Neighborhood {n_idx}...")
        neighborhood_consensus_model = get_consensus_model(neighborhood_models, device)

        # 3. Combine the test datasets from all agents in the neighborhood
        neighborhood_test_datasets = [loader.dataset for loader in neighborhood_test_loaders]
        combined_test_dataset = ConcatDataset(neighborhood_test_datasets)

        # We need a representative train loader for kNN and linear eval.
        # We can create one from the combined test dataset.
        combined_train_loader_for_eval = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=True)
        combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)

        print(
            f"Evaluating Neighborhood {n_idx} consensus model on combined test set ({len(combined_test_dataset)} samples)...")

        # 4. Run the standard evaluation functions
        knn_acc = knn_evaluation(
            model=neighborhood_consensus_model,
            train_loader=combined_train_loader_for_eval,
            test_loader=combined_test_loader,
            device=device,
            k=KNN_K,
            temperature=KNN_TEMPERATURE
        )
        wandb.summary[f"neighborhood_{n_idx}_final_knn_accuracy"] = knn_acc

        lin_acc = linear_evaluation(
            model=neighborhood_consensus_model,
            proj_output_dim=proj_input_dim,
            train_loader=combined_train_loader_for_eval,
            test_loader=combined_test_loader,
            epochs=eval_epochs,
            device=device
        )
        wandb.summary[f"neighborhood_{n_idx}_final_linear_accuracy"] = lin_acc