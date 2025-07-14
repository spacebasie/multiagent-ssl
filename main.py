# main.py

"""
Main script to run the VICReg self-supervised learning pipeline for CIFAR-10.
This is a simplified, robust version for federated and centralized runs.
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


def agent_update(agent_model, agent_dataloader, local_epochs, criterion, device):
    """
    Performs the local training for a single agent in a federated setting.
    """
    optimizer = torch.optim.SGD(agent_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    agent_model.train()
    agg_loss_dict = {}

    for _ in range(local_epochs):
        for batch in agent_dataloader:
            # Unpack for CIFAR-10 pre-training: ((view1, view2), label)
            (x0, x1), _ = batch
            x0, x1 = x0.to(device), x1.to(device)
            z0 = agent_model(x0)
            z1 = agent_model(x1)
            loss_dict = criterion(z0, z1)
            loss = loss_dict["loss"]

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                agg_loss_dict[k] = agg_loss_dict.get(k, 0.0) + v.item()

    num_batches = len(agent_dataloader) * local_epochs
    for k in agg_loss_dict:
        agg_loss_dict[k] /= num_batches
    return agg_loss_dict


def aggregate_models(agent_models):
    """Averages the weights of the agent models."""
    global_state_dict = copy.deepcopy(agent_models[0].state_dict())
    for key in global_state_dict.keys():
        for i in range(1, len(agent_models)):
            global_state_dict[key] += agent_models[i].state_dict()[key]
        global_state_dict[key] = torch.div(global_state_dict[key], len(agent_models))
    return global_state_dict


def train_one_epoch_centralized(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Handles the training logic for a single epoch in a centralized setting.
    """
    model.train()
    agg_loss_dict = {}
    for batch in dataloader:
        # Unpack for CIFAR-10 pre-training: ((view1, view2), label)
        (x0, x1), _ = batch
        x0, x1 = x0.to(device), x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss_dict = criterion(z0, z1)
        loss = loss_dict["loss"]

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            agg_loss_dict[k] = agg_loss_dict.get(k, 0.0) + v.item()

    scheduler.step()
    num_batches = len(dataloader)
    for k in agg_loss_dict:
        agg_loss_dict[k] /= num_batches
    return agg_loss_dict


def parse_arguments():
    """Parses command-line arguments for both modes."""
    parser = argparse.ArgumentParser(description="Unified VICReg Training Pipeline with W&B")
    parser.add_argument('--num_agents', type=int, default=config.NUM_AGENTS)
    parser.add_argument('--comm_rounds', type=int, default=config.COMMUNICATION_ROUNDS)
    parser.add_argument('--local_epochs', type=int, default=config.LOCAL_EPOCHS)
    parser.add_argument('--alpha', type=float, default=config.NON_IID_ALPHA)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--eval_every', type=int, default=config.EVAL_EVERY)
    return parser.parse_args()


def main():
    """Main function to execute the pipeline."""
    args = parse_arguments()
    wandb.init(project="cifar10-runs", config=args) # Using a new project for clarity
    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    pretrain_dataloader, train_loader_eval, test_loader_eval = get_dataloaders(
        dataset_path=config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        input_size=config.INPUT_SIZE
    )

    backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
    model = VICReg(backbone, proj_input_dim=config.PROJECTION_INPUT_DIM, proj_hidden_dim=config.PROJECTION_HIDDEN_DIM, proj_output_dim=config.PROJECTION_OUTPUT_DIM).to(device)
    criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)

    if args.num_agents == 1:
        wandb.run.name = f"centralized_epochs_{args.epochs}"
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        for epoch in range(args.epochs):
            loss_dict = train_one_epoch_centralized(model, pretrain_dataloader, optimizer, scheduler, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss_dict['loss']:.4f}")
            wandb.log({f"train/{k}": v for k, v in loss_dict.items()}, step=epoch + 1)
            if (epoch + 1) % args.eval_every == 0:
                knn_acc = knn_evaluation(model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)
                wandb.log({"eval/knn_accuracy": knn_acc}, step=epoch + 1)
    else:
        wandb.run.name = f"federated_agents_{args.num_agents}_alpha_{args.alpha}"
        agent_datasets = split_data(pretrain_dataloader.dataset, args.num_agents, args.alpha)
        agent_dataloaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) for ds in agent_datasets]
        for round_num in range(args.comm_rounds):
            print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
            agent_models = []
            round_agg_loss = {}
            for i in range(args.num_agents):
                agent_model = copy.deepcopy(model).to(device)
                loss_dict = agent_update(agent_model, agent_dataloaders[i], args.local_epochs, criterion, device)
                agent_models.append(agent_model)
                for k, v in loss_dict.items():
                    round_agg_loss[k] = round_agg_loss.get(k, 0.0) + v
            for k in round_agg_loss:
                round_agg_loss[k] /= args.num_agents
            wandb.log({f"train/avg_{k}": v for k, v in round_agg_loss.items()}, step=round_num + 1)
            global_state_dict = aggregate_models(agent_models)
            model.load_state_dict(global_state_dict)
            if (round_num + 1) % args.eval_every == 0:
                knn_acc = knn_evaluation(model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)
                wandb.log({"eval/knn_accuracy": knn_acc}, step=round_num + 1)

    print("\n--- Training Finished ---")
    final_linear_acc = linear_evaluation(model, config.PROJECTION_INPUT_DIM, train_loader_eval, test_loader_eval, config.EVAL_EPOCHS, device)
    final_knn_acc = knn_evaluation(model, train_loader_eval, test_loader_eval, device, config.KNN_K, config.KNN_TEMPERATURE)
    wandb.summary["final_linear_accuracy"] = final_linear_acc
    wandb.summary["final_knn_accuracy"] = final_knn_acc
    wandb.finish()


if __name__ == '__main__':
    main()
