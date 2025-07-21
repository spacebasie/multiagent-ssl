# decentralized_training.py

import copy
from torch.utils.data import DataLoader
from evaluate import linear_evaluation
from training import agent_update
from network import gossip_average

def decentralized_personalized_training(
    agent_models,
    agent_dataloaders,
    adj_matrix,
    criterion,
    device,
    test_loaders,  # List of domain-specific test loaders, one per agent
    comm_rounds,
    local_epochs,
    eval_every,
    wandb=None
):
    """
    Decentralized training with per-agent domain adaptation and personalized evaluation.
    Each agent is evaluated on its own domain-specific test set.
    """
    num_agents = len(agent_models)
    for round_num in range(comm_rounds):
        print(f"\n--- Round {round_num + 1}/{comm_rounds} ---")
        # Local training
        for i in range(num_agents):
            if len(agent_dataloaders[i].dataset) > 0:
                agent_update(agent_models[i], agent_dataloaders[i], local_epochs, criterion, device)
        # Gossip averaging
        agent_models = gossip_average(agent_models, adj_matrix)
        # Optional: log training metrics here

    # Personalized evaluation for each agent
    results = {}
    for i in range(num_agents):
        acc = linear_evaluation(
            agent_models[i],
            agent_models[i].projection_head.input_dim,
            agent_dataloaders[i],  # Use agent's own train loader
            test_loaders[i],       # Use agent's own test loader
            epochs=50,
            device=device
        )
        print(f"Agent {i} | Personalized Linear Test Accuracy: {acc:.2f}%")
        results[f"agent_{i}_linear_acc"] = acc
        if wandb:
            wandb.log({f"agent_{i}/linear_accuracy": acc})
    return results