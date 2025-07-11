# main.py

"""
Main script to run the VICReg self-supervised learning pipeline with Weights & Biases.
This script is unified to handle both:
1. Centralized benchmark runs (set --num_agents 1).
2. Federated/Multi-agent simulations (set --num_agents > 1).
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import copy
import argparse
import wandb

# Import from our modularized files
import config
from model import VICReg, VICRegLoss
from data import get_dataloaders
from data_splitter import split_data
from evaluate import linear_evaluation, knn_evaluation


# --- Federated Learning Functions ---

def agent_update(agent_model, agent_dataloader, local_epochs, device):
    """
    Performs the local training for a single agent in a federated setting.
    """
    criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)
    # Note: Federated settings often use simpler optimizers per round.
    optimizer = torch.optim.SGD(agent_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    agent_model.train()

    total_loss = 0
    num_batches = 0
    for _ in range(local_epochs):
        for batch in agent_dataloader:
            (x0, x1), _, _ = batch
            x0, x1 = x0.to(device), x1.to(device)
            z0 = agent_model(x0)
            z1 = agent_model(x1)
            loss = criterion(z0, z1)

            if torch.isnan(loss):
                print(f"  Warning: NaN loss detected for an agent. Skipping update.")
                continue

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
        for i in range(1, len(agent_models)):
            global_state_dict[key] += agent_models[i].state_dict()[key]
        global_state_dict[key] = torch.div(global_state_dict[key], len(agent_models))
    return global_state_dict


# --- Centralized Training Function ---

def train_one_epoch_centralized(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Handles the training logic for a single epoch in a centralized setting.
    """
    model.train()
    total_loss = 0
    for batch in dataloader:
        (x0, x1), _, _ = batch
        x0, x1 = x0.to(device), x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)

        if torch.isnan(loss):
            print(f"Warning: NaN loss detected. Skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(dataloader)


def parse_arguments():
    """Parses command-line arguments for both modes."""
    parser = argparse.ArgumentParser(description="Unified VICReg Training Pipeline with W&B")
    # General arguments
    parser.add_argument('--save_weights', action='store_true', help='If set, saves the final model backbone weights.')
    parser.add_argument('--wandb_project', type=str, default="ssl-experiments", help="W&B project name.")
    parser.add_argument('--wandb_entity', type=str, default=None, help="Your W&B entity (username or team).")

    # Federated Learning arguments
    parser.add_argument('--num_agents', type=int, default=config.NUM_AGENTS, help="Number of agents. Set to 1 for centralized training.")
    parser.add_argument('--comm_rounds', type=int, default=config.COMMUNICATION_ROUNDS, help="Communication rounds for FL.")
    parser.add_argument('--local_epochs', type=int, default=config.LOCAL_EPOCHS, help="Local epochs per agent per round for FL.")
    parser.add_argument('--alpha', type=float, default=config.NON_IID_ALPHA, help="Dirichlet alpha for non-IID data split.")

    # Centralized Benchmark arguments
    parser.add_argument('--epochs', type=int, default=config.BENCHMARK_EPOCHS, help="Total training epochs for centralized run.")
    parser.add_argument('--eval_every', type=int, default=25, help="Evaluate every N epochs/rounds.")

    return parser.parse_args()


def main():
    """Main function to execute the pipeline."""
    args = parse_arguments()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    # --- Data Preparation ---
    pretrain_dataloader, train_loader_eval, test_loader_eval = get_dataloaders(
        dataset_name=config.DATASET_NAME,
        dataset_path=config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        input_size=config.INPUT_SIZE
    )

    # --- Model Initialization ---
    backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
    model = VICReg(backbone, proj_input_dim=config.PROJECTION_INPUT_DIM, proj_hidden_dim=config.PROJECTION_HIDDEN_DIM, proj_output_dim=config.PROJECTION_OUTPUT_DIM).to(device)
    wandb.watch(model, log="all", log_freq=200)
    criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)

    # ==================================================================
    # --- BRANCHING LOGIC: Centralized vs. Federated ---
    # ==================================================================

    if args.num_agents == 1:
        # --- CENTRALIZED BENCHMARK RUN ---
        wandb.run.name = f"centralized_{config.DATASET_NAME}_epochs_{args.epochs}"
        print(f"--- Starting Centralized Benchmark on {config.DATASET_NAME} ---")

        optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        max_knn_accuracy = 0.0

        for epoch in range(args.epochs):
            avg_loss = train_one_epoch_centralized(model, pretrain_dataloader, optimizer, scheduler, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                knn_acc = knn_evaluation(model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)
                if knn_acc > max_knn_accuracy:
                    max_knn_accuracy = knn_acc
                    wandb.summary["best_knn_accuracy"] = max_knn_accuracy
                    wandb.summary["best_knn_epoch"] = epoch + 1
                wandb.log({"eval/knn_accuracy": knn_acc, "eval/max_knn_accuracy": max_knn_accuracy, "epoch": epoch + 1})

            wandb.log({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0], "epoch": epoch + 1})

    else:
        # --- FEDERATED LEARNING RUN ---
        wandb.run.name = f"federated_agents_{args.num_agents}_alpha_{args.alpha}"
        print(f"--- Starting Federated Learning Simulation ---")

        agent_datasets = split_data(pretrain_dataloader.dataset, args.num_agents, args.alpha)
        agent_dataloaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) for ds in agent_datasets]

        for round_num in range(args.comm_rounds):
            print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
            agent_models = []
            round_losses = []
            for i in range(args.num_agents):
                agent_model = copy.deepcopy(model).to(device)
                avg_loss = agent_update(agent_model, agent_dataloaders[i], args.local_epochs, device)
                agent_models.append(agent_model)
                if avg_loss > 0: round_losses.append(avg_loss)

            if round_losses:
                avg_round_loss = sum(round_losses) / len(round_losses)
                wandb.log({"train/avg_agent_loss": avg_round_loss, "communication_round": round_num + 1})

            global_state_dict = aggregate_models(agent_models)
            model.load_state_dict(global_state_dict)

            if (round_num + 1) % args.eval_every == 0:
                knn_acc = knn_evaluation(model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)
                wandb.log({"eval/knn_accuracy": knn_acc, "communication_round": round_num + 1})

    # --- Final Actions for both modes ---
    print("\n--- Training Finished ---")
    print("\n--- Starting Final Evaluation ---")
    final_linear_acc = linear_evaluation(model, config.PROJECTION_INPUT_DIM, train_loader_eval, test_loader_eval, config.EVAL_EPOCHS, device)
    wandb.summary["final_linear_accuracy"] = final_linear_acc

    if args.save_weights:
        model_save_path = f"run_{wandb.run.id}_backbone.pth"
        print(f"\n--- Saving final model weights to {model_save_path} ---")
        torch.save(model.backbone.state_dict(), model_save_path)
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_file(model_save_path)
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    main()
