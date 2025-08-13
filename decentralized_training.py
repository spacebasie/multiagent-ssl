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
    learning_rate
):
    """
    Decentralized training with per-agent domain adaptation and personalized evaluation.
    Each agent is evaluated on its own domain-specific test set.

    Args:
        agent_models (list): List of agent models.
        agent_train_dataloaders (list): List of agent-specific training dataloaders.
        agent_test_dataloaders (list): List of agent-specific test dataloaders.
        adj_matrix (np.ndarray): The network adjacency matrix.
        criterion: The loss function.
        device (str): The device to train on.
        comm_rounds (int): Total number of communication rounds.
        local_epochs (int): Number of local training epochs per round.
        eval_every (int): Frequency of evaluation rounds.
        proj_input_dim (int): The input dimension of the projection head.
        eval_epochs (int): Number of epochs for linear evaluation.
    """
    num_agents = len(agent_models)
    for round_num in range(comm_rounds):
        print(f"\n--- Round {round_num + 1}/{comm_rounds} ---")
        round_agg_loss = {}
        # Local training for each agent
        for i in range(num_agents):
            if len(agent_train_dataloaders[i].dataset) > 0:
                print(f"Training Agent {i}...")
                loss_dict = agent_update(agent_models[i], agent_train_dataloaders[i], local_epochs, criterion, device, learning_rate)
                if loss_dict:
                    for k, v in loss_dict.items():
                        round_agg_loss[k] = round_agg_loss.get(k, 0.0) + v

        if round_agg_loss:
            num_successful_agents = num_agents  # Assuming all agents train successfully
            for k in round_agg_loss:
                round_agg_loss[k] /= num_successful_agents
            wandb.log({f"train/avg_{k}": v for k, v in round_agg_loss.items()}, step=round_num + 1)

        # Gossip averaging communication step
        print("Performing gossip averaging...")
        agent_models = gossip_average(agent_models, adj_matrix)

        # --- Personalized Evaluation ---
        if (round_num + 1) % eval_every == 0:
            print(f"\n--- Evaluating Personalized Models at Round {round_num + 1} ---")
            total_acc = 0
            for i in range(num_agents):
                # Evaluate each agent's model on its OWN domain-specific test data
                acc = knn_evaluation(
                    model=agent_models[i],
                    train_loader=agent_train_dataloaders[i],
                    test_loader=agent_test_dataloaders[i],
                    device=device,
                    k=KNN_K,
                    temperature=KNN_TEMPERATURE
                )
                total_acc += acc
                wandb.log({f"eval/agent_{i}_knn_accuracy": acc}, step=round_num + 1)

            avg_acc = total_acc / num_agents
            print(f"Average Personalized KNN Test Accuracy: {avg_acc:.2f}%")
            wandb.log({"eval/avg_personalized_accuracy": avg_acc}, step=round_num + 1)

            # Also evaluate the consensus model on a global test set for comparison
            print("\n--- Evaluating Consensus Model on Global Test Set ---")
            # Note: This requires a global test loader, which we can create in main.py
            # For now, we'll just log the personalized results which are the focus of this path.
            consensus_model = get_consensus_model(agent_models, device)
            if consensus_model:
                # You would need a global test loader here if you want to run this
                pass # knn_acc = knn_evaluation(consensus_model, ...)

    print("\n--- Final Personalized Evaluation ---")
    final_results = {}
    lin_final_results = {}
    for i in range(num_agents):
        acc = knn_evaluation(
            model=agent_models[i],
            train_loader=agent_train_dataloaders[i],
            test_loader=agent_test_dataloaders[i],
            device=device,
            k = KNN_K,
            temperature = KNN_TEMPERATURE
        )
        final_results[f"agent_{i}_final_knn_acc"] = acc
        wandb.summary[f"agent_{i}_final_knn_accuracy"] = acc

        lin_acc = linear_evaluation(
            model=agent_models[i],
            proj_output_dim=proj_input_dim,
            train_loader=agent_train_dataloaders[i],
            test_loader=agent_test_dataloaders[i],
            epochs=eval_epochs,
            device=device
        )
        lin_final_results[f"agent_{i}_final_linear_acc"] = lin_acc
        wandb.summary[f"agent_{i}_final_linear_accuracy"] = lin_acc

        # Uncomment this to check when I validate the new implementation
        plot_tsne(
            model=agent_models[i],
            test_loader=agent_test_dataloaders[i],
            device=device,
            plot_title=f"t-SNE for Agent {i}"
        )
        plot_pca(model=agent_models[i],
                 test_loader=agent_test_dataloaders[i],
                 device=device,
                 plot_title=f"PCA for Agent {i}")

    avg_final_acc = sum(final_results.values()) / len(final_results)
    avg_final_lin_acc = sum(lin_final_results.values()) / len(lin_final_results)
    wandb.summary["average_final_personalized_accuracy"] = avg_final_acc
    wandb.summary["average_final_linear_accuracy"] = avg_final_lin_acc
    print(f"\nAverage Final Personalized Accuracy: {avg_final_acc:.2f}%")
    print(f"Average Final Linear Evaluation Accuracy: {avg_final_lin_acc:.2f}%")