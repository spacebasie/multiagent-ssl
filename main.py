# main.py

"""
Main script to run the VICReg self-supervised learning pipeline.
Supports both centralized and federated training modes.

Run `python main.py --help` to see all available options.
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import copy
import argparse
import matplotlib.pyplot as plt

# Import from our modularized files
import config
from model import VICReg, VICRegLoss
from data import get_dataloaders
from data_splitter import split_data
from evaluate import linear_evaluation, knn_evaluation
from lightly.transforms.vicreg_transform import VICRegTransform


# --- Helper Functions for Federated Learning ---

def agent_update(agent_model, agent_dataloader, local_epochs, device):
    """Performs the local training for a single agent."""
    criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)
    optimizer = torch.optim.SGD(agent_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    agent_model.train()
    for _ in range(local_epochs):
        for batch in agent_dataloader:
            x0, x1 = batch[0]
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = agent_model(x0), agent_model(z1)
            loss = criterion(z0, z1)
            if torch.isnan(loss): return
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def federated_average(agent_models):
    """Averages the weights of the agent models."""
    global_state_dict = copy.deepcopy(agent_models[0].state_dict())
    for key in global_state_dict.keys():
        for i in range(1, len(agent_models)):
            global_state_dict[key] += agent_models[i].state_dict()[key]
        global_state_dict[key] = torch.div(global_state_dict[key], len(agent_models))
    return global_state_dict


# --- Main Training Logic ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="VICReg Training and Evaluation Pipeline")

    parser.add_argument('--mode', type=str, default=config.TRAINING_MODE, choices=['centralized', 'federated'],
                        help='Training mode: centralized or federated.')

    # Federated specific arguments
    parser.add_argument('--num_agents', type=int, default=config.NUM_AGENTS,
                        help='Number of agents for federated learning.')
    parser.add_argument('--comm_rounds', type=int, default=config.COMMUNICATION_ROUNDS,
                        help='Number of communication rounds for federated learning.')
    parser.add_argument('--alpha', type=float, default=config.NON_IID_ALPHA,
                        help='Alpha for non-IID data split.')
    parser.add_argument('--local_epochs', type=int, default=config.EPOCHS,
                        help='Number of local epochs per round in federated mode.')

    # Centralized specific arguments
    parser.add_argument('--centralized_epochs', type=int, default=config.CENTRALIZED_EPOCHS,
                        help='Total number of epochs for centralized training.')

    return parser.parse_args()


def main():
    """Main function to execute the full pipeline based on the selected mode."""
    args = parse_arguments()
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Selected Mode: {args.mode}")

    # --- Model Initialization ---
    backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
    global_model = VICReg(
        backbone,
        proj_input_dim=config.PROJECTION_INPUT_DIM,
        proj_hidden_dim=config.PROJECTION_HIDDEN_DIM,
        proj_output_dim=config.PROJECTION_OUTPUT_DIM
    ).to(device)

    # --- Execute Training based on Mode ---
    if args.mode == 'centralized':
        # --- Centralized Training ---
        print("\n--- Starting Centralized Training ---")
        pretrain_loader, _, _ = get_dataloaders(config.BATCH_SIZE, config.NUM_WORKERS, config.INPUT_SIZE,
                                                config.DATASET_PATH)

        criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

        for epoch in range(args.centralized_epochs):
            total_loss = 0
            for batch in pretrain_loader:
                x0, x1 = batch[0]
                x0, x1 = x0.to(device), x1.to(device)
                z0, z1 = global_model(x0), global_model(z1)
                loss = criterion(z0, z1)
                total_loss += loss.detach()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(pretrain_loader)
            print(f"Epoch: {epoch:02}, Loss: {avg_loss:.5f}")

    elif args.mode == 'federated':
        # --- Federated Training ---
        print("\n--- Starting Federated Training ---")
        transform_vicreg = VICRegTransform(input_size=config.INPUT_SIZE)
        full_train_dataset = torchvision.datasets.CIFAR10(root=config.DATASET_PATH, download=True, train=True,
                                                          transform=transform_vicreg)

        agent_datasets = split_data(full_train_dataset, args.num_agents, args.alpha)
        agent_dataloaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
                             for ds in agent_datasets]

        for round_num in range(args.comm_rounds):
            print(f"\n--- Communication Round {round_num + 1}/{args.comm_rounds} ---")
            agent_models = []
            for i in range(args.num_agents):
                print(f"  Training Agent {i + 1}/{args.num_agents}...")
                agent_model = copy.deepcopy(global_model).to(device)
                agent_update(agent_model, agent_dataloaders[i], args.local_epochs, device)
                agent_models.append(agent_model)

            print("  Averaging agent models on the server...")
            global_state_dict = federated_average(agent_models)
            global_model.load_state_dict(global_state_dict)

    # --- Save and Evaluate the Final Model ---
    model_save_path = f"{args.mode}_{config.MODEL_SAVE_PATH}"
    print(f"\n--- Training Finished. Saving model to {model_save_path} ---")
    torch.save(global_model.backbone.state_dict(), model_save_path)

    print("\n--- Starting Final Evaluation on Global Model ---")
    _, train_loader_eval, test_loader_eval = get_dataloaders(config.BATCH_SIZE, config.NUM_WORKERS, config.INPUT_SIZE,
                                                             config.DATASET_PATH)
    linear_evaluation(global_model, config.PROJECTION_INPUT_DIM, train_loader_eval, test_loader_eval,
                      config.EVAL_EPOCHS, device)
    knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)


if __name__ == '__main__':
    main()
