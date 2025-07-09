# main.py

"""
Main script to run the VICReg self-supervised learning pipeline with Weights & Biases.
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import copy
import argparse
from datetime import datetime
import wandb # Import wandb

# Import from our modularized files
import config
from model import VICReg, VICRegLoss
from data_splitter import split_data
from evaluate import linear_evaluation, knn_evaluation
from lightly.transforms.vicreg_transform import VICRegTransform


def agent_update(agent_model, agent_dataloader, local_epochs, device):
    """
    Performs the local training for a single agent and returns the average loss.
    """
    criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)
    optimizer = torch.optim.SGD(agent_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    agent_model.train()

    total_loss = 0
    num_batches = 0
    for _ in range(local_epochs):
        for batch in agent_dataloader:
            x0, x1 = batch[0]
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = agent_model(x0), agent_model(x1)
            loss = criterion(z0, z1)

            if torch.isnan(loss):
                print(f"  Warning: NaN loss detected for an agent. Skipping update.")
                continue # Skip this batch if loss is NaN

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def aggregate_models(agent_models):
    """Averages the weights of the agent models."""
    global_state_dict = copy.deepcopy(agent_models[0].state_dict())
    for key in global_state_dict.keys():
        # Sum the weights from all other agents
        for i in range(1, len(agent_models)):
            global_state_dict[key] += agent_models[i].state_dict()[key]
        # Divide by the number of agents to get the average
        global_state_dict[key] = torch.div(global_state_dict[key], len(agent_models))
    return global_state_dict


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Decentralized VICReg Training Pipeline with W&B")
    parser.add_argument('--num_agents', type=int, default=config.NUM_AGENTS)
    parser.add_argument('--comm_rounds', type=int, default=config.COMMUNICATION_ROUNDS)
    parser.add_argument('--local_epochs', type=int, default=config.LOCAL_EPOCHS)
    parser.add_argument('--alpha', type=float, default=config.NON_IID_ALPHA)
    parser.add_argument('--eval_epochs', type=int, default=config.EVAL_EPOCHS)
    parser.add_argument('--eval_every', type=int, default=5, help="Evaluate every N communication rounds.")
    parser.add_argument('--save_weights', action='store_true', help='If set, saves the final model backbone weights.')
    parser.add_argument('--wandb_project', type=str, default="multiagent-ssl", help="Multiagent VICReg")
    parser.add_argument('--wandb_entity', type=str, default=None, help="Stef-fri")
    return parser.parse_args()


def main():
    """Main function to execute the full pipeline."""
    args = parse_arguments()

    # --- W&B Initialization ---
    # Start a new run, tracking hyperparameters in the config
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args
    )
    # Use a descriptive name for the run
    wandb.run.name = f"agents_{args.num_agents}_alpha_{args.alpha}_rounds_{args.comm_rounds}_epochs_{args.local_epochs}"
    wandb.run.save()


    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print("--- Starting VICReg Experiment with W&B ---")
    print(f"Device: {device}, Agents: {args.num_agents}, Rounds: {args.comm_rounds}, Local Epochs: {args.local_epochs}")
    if args.num_agents > 1: print(f"Non-IID Alpha: {args.alpha}")

    # --- Data Preparation ---
    transform_vicreg = VICRegTransform(input_size=config.INPUT_SIZE)
    full_train_dataset = torchvision.datasets.CIFAR10(root=config.DATASET_PATH, download=True, train=True, transform=transform_vicreg)
    agent_datasets = split_data(full_train_dataset, args.num_agents, args.alpha)
    agent_dataloaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) for ds in agent_datasets]

    transform_eval = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset_eval = torchvision.datasets.CIFAR10(root=config.DATASET_PATH, train=True, transform=transform_eval)
    test_dataset_eval = torchvision.datasets.CIFAR10(root=config.DATASET_PATH, train=False, transform=transform_eval)
    train_loader_eval = DataLoader(train_dataset_eval, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader_eval = DataLoader(test_dataset_eval, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
    global_model = VICReg(backbone, proj_input_dim=config.PROJECTION_INPUT_DIM, proj_hidden_dim=config.PROJECTION_HIDDEN_DIM, proj_output_dim=config.PROJECTION_OUTPUT_DIM).to(device)
    wandb.watch(global_model, log="all", log_freq=100) # Watch model gradients

    # --- Training Loop ---
    for round_num in range(args.comm_rounds):
        print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
        agent_models = []
        round_losses = []
        for i in range(args.num_agents):
            agent_model = copy.deepcopy(global_model).to(device)
            avg_loss = agent_update(agent_model, agent_dataloaders[i], args.local_epochs, device)
            agent_models.append(agent_model)
            if avg_loss > 0: round_losses.append(avg_loss)

        # Log average training loss for the round
        if round_losses:
            avg_round_loss = sum(round_losses) / len(round_losses)
            wandb.log({"train/avg_agent_loss": avg_round_loss, "communication_round": round_num + 1})

        # Aggregate models
        global_state_dict = aggregate_models(agent_models)
        global_model.load_state_dict(global_state_dict)

        # Evaluate periodically
        if (round_num + 1) % args.eval_every == 0:
            print(f"\n--- Evaluating at Round {round_num + 1} ---")
            knn_acc = knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)
            # Log the kNN accuracy to W&B to create the learning curve
            wandb.log({"eval/knn_accuracy": knn_acc, "communication_round": round_num + 1})

    # --- Final Actions ---
    print("\n--- Training Finished ---")

    # Run final, detailed evaluations
    print("\n--- Starting Final Evaluation ---")
    final_linear_acc = linear_evaluation(global_model, config.PROJECTION_INPUT_DIM, train_loader_eval, test_loader_eval, config.EVAL_EPOCHS, device)
    final_knn_acc = knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)

    # Log final metrics to the W&B summary
    wandb.summary["final_linear_accuracy"] = final_linear_acc
    wandb.summary["final_knn_accuracy"] = final_knn_acc

    # Optionally save the model weights
    if args.save_weights:
        model_save_path = f"run_{wandb.run.id}_backbone.pth"
        print(f"\n--- Saving final model weights to {model_save_path} ---")
        torch.save(global_model.backbone.state_dict(), model_save_path)
        # Save model artifact to W&B
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_file(model_save_path)
        wandb.log_artifact(artifact)
    else:
        print("\n--- Skipping model weight saving as per configuration ---")

    # Finish the W&B run
    wandb.finish()


if __name__ == '__main__':
    main()
