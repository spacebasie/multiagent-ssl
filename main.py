# main.py

"""
Main script to run the VICReg self-supervised learning pipeline.
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import copy
import argparse
import wandb

# Import from our modularized files
import config
from model import VICReg, VICRegLoss
from data import get_dataloaders
from data_splitter import split_data, get_domain_shift_dataloaders
from evaluate import linear_evaluation, knn_evaluation
from network import set_network_topology, gossip_average
from training import agent_update, aggregate_models, get_consensus_model, train_one_epoch_centralized
from decentralized_training import decentralized_personalized_training
from lightly.transforms.vicreg_transform import VICRegTransform
import torchvision.transforms as T

class PreTransform(nn.Module):
    """
    Applies a custom 'pre_transform' before the main transform.
    This ensures the domain shift happens to the raw image *before* VICReg's augmentation.
    """
    def __init__(self, pre_transform, main_transform):
        super().__init__()
        self.pre_transform = pre_transform
        self.main_transform = main_transform

    def __call__(self, x):
        # 1. Apply the domain-specific transform (e.g., blur) to the raw image first.
        x = self.pre_transform(x)
        # 2. Then, apply the standard VICReg augmentation to the corrupted image.
        x = self.main_transform(x)
        return x

def parse_arguments():
    """Parses command-line arguments for both modes."""
    parser = argparse.ArgumentParser(description="Unified VICReg Training Pipeline with W&B")
    parser.add_argument('--mode', type=str, default='centralized', choices=['centralized', 'federated', 'decentralized'])
    parser.add_argument('--heterogeneity_type', type=str, default='label_skew',
                        choices=['label_skew', 'domain_shift'],
                        help='The type of data heterogeneity for decentralized mode.')
    parser.add_argument('--topology', type=str, default=config.NETWORK_TOPOLOGY, choices=['ring', 'fully_connected', 'random'])
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
    wandb.init(project="cifar10-runs", config=args)

    wandb.config.update({
        "lambda": config.LAMBDA,
        "mu": config.MU,
        "nu": config.NU,
        "knn_k": config.KNN_K,
        "knn_temperature": config.KNN_TEMPERATURE,
        "learning_rate": config.LEARNING_RATE,
    })

    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    num_classes = 100 if config.DATASET_NAME == 'cifar100' else 10
    wandb.config.update({"dataset": config.DATASET_NAME, "num_classes": num_classes})

    pretrain_dataloader, train_loader_eval, test_loader_eval = get_dataloaders(
        dataset_name=config.DATASET_NAME,
        dataset_path=config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        input_size=config.INPUT_SIZE
    )

    # Alternative backbone setup specifically for CIFAR-10
    # 1. Create a standard torchvision ResNet-18
    # resnet = torchvision.models.resnet18()
    # # 2. Modify it for CIFAR-10 as per the benchmark description
    # #    Replace the first 7x7 conv with a 3x3 conv
    # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # #    Remove the initial MaxPool layer
    # resnet.maxpool = nn.Identity()
    # # 3. The backbone is the modified ResNet without its final classification layer
    # backbone = nn.Sequential(*list(resnet.children())[:-1])

    # --- MODEL INITIALIZATION (Centralized and Federated) ---
    # resnet = torchvision.models.resnet18()
    # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = nn.Identity()
    # backbone = nn.Sequential(*list(resnet.children())[:-1])

    # Original backbone
    backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
    global_model = VICReg(backbone, proj_input_dim=config.PROJECTION_INPUT_DIM,
                          proj_hidden_dim=config.PROJECTION_HIDDEN_DIM,
                          proj_output_dim=config.PROJECTION_OUTPUT_DIM).to(device)

    criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)

    final_model_to_eval = None

    if args.mode == 'centralized':
        wandb.run.name = f"centralized_{config.DATASET_NAME}_epochs_{args.epochs}"
        optimizer = torch.optim.SGD(global_model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        # Use the warmup scheduler
        # scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=args.epochs)

        for epoch in range(args.epochs):
            loss_dict = train_one_epoch_centralized(global_model, pretrain_dataloader, optimizer, criterion, device)
            # Step the scheduler after the epoch is complete
            # scheduler.step()

            if not loss_dict:
                print(f"Epoch {epoch + 1}/{args.epochs} | Training unstable, all batches produced NaN loss. Stopping.")
                break

            print(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {loss_dict['loss']:.4f} | LR: {optimizer.param_groups[0]['lr']:.5f}")
            wandb.log({f"train/{k}": v for k, v in loss_dict.items()}, step=epoch + 1)
            wandb.log({"train/lr": optimizer.param_groups[0]['lr']}, step=epoch + 1)

            if (epoch + 1) % args.eval_every == 0:
                knn_acc = knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K,
                                         config.KNN_TEMPERATURE)
                wandb.log({"eval/knn_accuracy": knn_acc}, step=epoch + 1)
        final_model_to_eval = global_model

    elif args.mode == 'federated':
        # Federated learning logic
        wandb.run.name = f"federated_agents_{args.num_agents}_alpha_{args.alpha}"
        agent_datasets = split_data(pretrain_dataloader.dataset, args.num_agents, args.alpha)
        # 1. Filter out any agents that were assigned no data
        active_agent_datasets = [ds for ds in agent_datasets if len(ds) > 0]
        print(f"Data split resulted in {len(active_agent_datasets)} active agents out of {args.num_agents}.")
        # 2. Create DataLoaders only for active agents
        agent_dataloaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
                             for ds in active_agent_datasets]

        for round_num in range(args.comm_rounds):
            print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
            agent_models = []
            round_agg_loss = {}

            # 3. Iterate over the active dataloaders, not a fixed range
            for agent_dataloader in agent_dataloaders:
                agent_model = copy.deepcopy(global_model).to(device)
                loss_dict = agent_update(agent_model, agent_dataloader, args.local_epochs, criterion, device)

                if loss_dict:  # Check if the agent trained successfully
                    agent_models.append(agent_model)
                    for k, v in loss_dict.items():
                        round_agg_loss[k] = round_agg_loss.get(k, 0.0) + v

            if not agent_models:
                print(f"Round {round_num + 1} | All active agents failed to train. Stopping.")
                break

            # 4. Average losses and models over the number of agents that actually trained
            num_successful_agents = len(agent_models)
            for k in round_agg_loss:
                round_agg_loss[k] /= num_successful_agents
            wandb.log({f"train/avg_{k}": v for k, v in round_agg_loss.items()}, step=round_num + 1)

            global_state_dict = aggregate_models(agent_models)
            global_model.load_state_dict(global_state_dict)
            if (round_num + 1) % args.eval_every == 0:
                knn_acc = knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K,
                                         config.KNN_TEMPERATURE)
                wandb.log({"eval/knn_accuracy": knn_acc}, step=round_num + 1)

    elif args.mode == 'decentralized':
        wandb.run.name = f"decentralized_{args.topology}_agents_{args.num_agents}_alpha_{args.alpha}"
        print(f"--- Starting Decentralized P2P Simulation ({args.topology}) ---")

        # 1. Create the network topology
        adj_matrix = set_network_topology(args.topology, args.num_agents)

        # 2. Initialize N separate agent models
        agent_models = [copy.deepcopy(global_model).to(device) for _ in range(args.num_agents)]

        if args.heterogeneity_type == 'label_skew':
            agent_datasets = split_data(pretrain_dataloader.dataset, args.num_agents, args.alpha)
            agent_dataloaders = [
                DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) for ds in
                agent_datasets if len(ds) > 0]

            for round_num in range(args.comm_rounds):
                print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
                for i in range(len(agent_dataloaders)):
                    agent_update(agent_models[i], agent_dataloaders[i], args.local_epochs, criterion, device)
                agent_models = gossip_average(agent_models, adj_matrix)

                if (round_num + 1) % args.eval_every == 0:
                    consensus_model = get_consensus_model(agent_models, device)
                    if consensus_model:
                        knn_acc = knn_evaluation(consensus_model, train_loader_eval, test_loader_eval, device,
                                                 config.KNN_K, config.KNN_TEMPERATURE)
                        wandb.log({"eval/consensus_knn_accuracy": knn_acc}, step=round_num + 1)

            final_model_to_eval = get_consensus_model(agent_models, device)

            # --- Sub-mode: Domain Shift with Personalized Evaluation ---
        elif args.heterogeneity_type == 'domain_shift':
            vicreg_transform = VICRegTransform(input_size=config.INPUT_SIZE)

            domain_shifts = [
                T.Compose([]),  # Base (clean) - does nothing to the PIL image
                T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
                T.RandomRotation(degrees=90),
                T.ColorJitter(brightness=0.5, contrast=0.5),
                T.RandomPerspective(distortion_scale=0.5, p=1.0),  # Replaced RandomErasing
            ]

            # Cycle through the defined shifts to cover all agents
            final_domain_shifts = []
            for i in range(args.num_agents):
                shift_to_apply = domain_shifts[i % len(domain_shifts)]
                final_domain_shifts.append(shift_to_apply)

            agent_specific_transforms = [
                PreTransform(pre_transform=ds, main_transform=vicreg_transform) for ds in final_domain_shifts
            ]

            agent_train_dataloaders, agent_test_dataloaders = get_domain_shift_dataloaders(
                train_dataset=pretrain_dataloader.dataset,
                test_dataset=test_loader_eval.dataset,
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                num_agents=args.num_agents,
                agent_transforms=agent_specific_transforms
            )

            decentralized_personalized_training(
                agent_models=agent_models, agent_train_dataloaders=agent_train_dataloaders,
                agent_test_dataloaders=agent_test_dataloaders, adj_matrix=adj_matrix,
                criterion=criterion, device=device, comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs, eval_every=args.eval_every,
                proj_input_dim=config.PROJECTION_INPUT_DIM, eval_epochs=config.EVAL_EPOCHS
            )
            # Final evaluation is handled inside the personalized loop for this mode

        # # 3. Distribute data
        # agent_datasets = split_data(pretrain_dataloader.dataset, args.num_agents, args.alpha)
        # agent_dataloaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        #                      for ds in agent_datasets]
        #
        # for round_num in range(args.comm_rounds):
        #     print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
        #
        #     # 4. Local training step for each agent on its own model
        #     for i in range(args.num_agents):
        #         if len(agent_dataloaders[i].dataset) > 0:
        #             agent_update(agent_models[i], agent_dataloaders[i], args.local_epochs, criterion, device)
        #
        #     # 5. Peer-to-peer communication step
        #     agent_models = gossip_average(agent_models, adj_matrix)
        #
        #     # 6. Evaluation (on the average/consensus model)
        #     if (round_num + 1) % args.eval_every == 0:
        #         consensus_model = get_consensus_model(agent_models, device)
        #         if consensus_model:
        #             knn_acc = knn_evaluation(consensus_model, train_loader_eval, test_loader_eval, device, config.KNN_K,
        #                                      config.KNN_TEMPERATURE)
        #             wandb.log({"eval/knn_accuracy": knn_acc}, step=round_num + 1)


    if final_model_to_eval:
        print("\n--- Training Finished ---")
        final_linear_acc = linear_evaluation(final_model_to_eval, config.PROJECTION_INPUT_DIM, train_loader_eval,
                                             test_loader_eval, config.EVAL_EPOCHS, device)
        final_knn_acc = knn_evaluation(final_model_to_eval, train_loader_eval, test_loader_eval, device, config.KNN_K,
                                       config.KNN_TEMPERATURE)
        wandb.summary["final_linear_accuracy"] = final_linear_acc
        wandb.summary["final_knn_accuracy"] = final_knn_acc
    wandb.finish()


if __name__ == '__main__':
    main()
