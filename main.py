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
from data_splitter import split_data, get_domain_shift_dataloaders, split_train_test_data_personalized
from evaluate import linear_evaluation, knn_evaluation, plot_tsne, plot_pca
from network import set_network_topology, gossip_average, set_hierarchical_topology
from training import agent_update, aggregate_models, get_consensus_model, train_one_epoch_centralized
from decentralized_training import decentralized_personalized_training, evaluate_neighborhood_consensus
from custom_datasets import (get_officehome_train_test_loaders, get_officehome_domain_split_loaders_personalized,
                             get_officehome_domain_split_loaders_global, get_officehome_hierarchical_loaders, get_public_dataloader)
from combo_training import alignment_collaborative_training, evaluate_combo_model, final_combo_evaluation
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
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'office_home'])
    parser.add_argument('--heterogeneity_type', type=str, default='label_skew',
                        choices=['label_skew', 'label_skew_personalized', 'domain_shift', 'office_random', 'office_domain_split', 'office_hierarchical', 'combo_domain'],
                        help='The type of data heterogeneity for decentralized mode.')
    parser.add_argument('--topology', type=str, default=config.NETWORK_TOPOLOGY, choices=['ring', 'fully_connected', 'random', 'disconnected'],
                        help='Network topology for decentralized mode.')
    parser.add_argument('--num_agents', type=int, default=config.NUM_AGENTS)
    parser.add_argument('--comm_rounds', type=int, default=config.COMMUNICATION_ROUNDS)
    parser.add_argument('--local_epochs', type=int, default=config.LOCAL_EPOCHS)
    parser.add_argument('--alpha', type=float, default=config.NON_IID_ALPHA)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--eval_every', type=int, default=config.EVAL_EVERY)
    parser.add_argument('--num_classes', type=int, default=0,
                        help='Number of classes to use from the dataset (0 for all).')
    # parser.add_argument('--num_neighborhoods', type=int, default=2,
    #                     help='Number of neighborhoods for hierarchical setup.')
    # parser.add_argument('--agents_per_neighborhood', type=int, default=4,
    #                     help='Number of agents per neighborhood for hierarchical setup.')
    parser.add_argument('--alignment_strength', type=int, default=1,
                        help='Strength of the alignment regularization for combo_domain heterogeneity.')
    return parser.parse_args()


def main():
    """Main function to execute the pipeline."""
    args = parse_arguments()
    if args.heterogeneity_type == 'office_hierarchical':
        args.num_agents = args.num_neighborhoods * args.agents_per_neighborhood
    if args.dataset == 'cifar10':
        wandb.init(project="cifar10-runs", config=args)
    elif args.dataset == 'office_home':
        wandb.init(project="office_home_runs", config=args)

    wandb.config.update({
        "lambda": config.LAMBDA,
        "mu": config.MU,
        "nu": config.NU,
        "knn_k": config.KNN_K,
        "knn_temperature": config.KNN_TEMPERATURE,
        "learning_rate": config.LEARNING_RATE,
    })

    if args.mode == 'centralized':
        batch_size = config.BATCH_SIZE
    else:
        # For federated/decentralized, divide the global batch size among agents
        batch_size = config.BATCH_SIZE // args.num_agents
        # Enforce a minimum batch size for stability
        if batch_size < 8:
            print(f"Warning: Calculated batch size ({batch_size}) is too small. Setting to minimum of 8.")
            batch_size = 8

        # Update args to log the actual batch size used
    args.batch_size = batch_size

    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    dataset_config = None
    pretrain_dataloader = None
    agent_train_dataloaders = None
    agent_test_dataloaders = None

    if args.dataset == 'office_home':
        wandb.config.update({"dataset": "office_home"})
        # Define the two separate transforms needed for OfficeHome
        vicreg_transform_officehome = VICRegTransform(input_size=224)
        eval_transform_officehome = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if args.mode == 'federated':
            print("Using OfficeHome domain split for federated learning.")
            # Use the new function to get domain-expert dataloaders
            agent_dataloaders, train_loader_eval, test_loader_eval = get_officehome_domain_split_loaders_global(
                root_dir="datasets/OfficeHomeDataset",
                num_agents=args.num_agents,
                batch_size=args.batch_size,
                num_workers=config.NUM_WORKERS,
                train_transform=vicreg_transform_officehome,
                eval_transform=eval_transform_officehome,
                num_classes=args.num_classes
            )
        else:
            agent_train_dataloaders, agent_test_dataloaders, train_loader_eval, test_loader_eval = get_officehome_train_test_loaders(
                root_dir="datasets/OfficeHomeDataset",
                num_agents=args.num_agents,
                batch_size=args.batch_size,
                num_workers=config.NUM_WORKERS,
                train_transform=vicreg_transform_officehome,
                eval_transform=eval_transform_officehome,
                num_classes=args.num_classes
            )

        if args.mode == 'centralized':
            # For centralized mode, the pretrain_dataloader uses the full training set with VICReg transforms
            pretrain_dataset = train_loader_eval.dataset
            pretrain_dataset.transform = vicreg_transform_officehome
            pretrain_dataloader = DataLoader(
                pretrain_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=config.NUM_WORKERS
            )
    else:  # This block handles cifar10, cifar100, etc.
        num_classes = 100 if args.dataset == 'cifar100' else 10
        wandb.config.update({"dataset": args.dataset, "num_classes": num_classes})

        pretrain_dataloader, train_loader_eval, test_loader_eval, dataset_config = get_dataloaders(
            dataset_name=args.dataset,
            dataset_path=config.DATASET_PATH,
            batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS
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
        wandb.run.name = f"federated_agents_{args.num_agents}_dataset_{args.dataset}"
        # Conditionally prepare agent dataloaders based on the dataset
        if args.dataset == 'office_home':
            print("Using OfficeHome domain split for federated learning.")
        else:
            # Original, working logic for CIFAR datasets using label skew
            print(f"Creating label skew (alpha={args.alpha}) splits for {args.dataset}.")
            agent_datasets = split_data(pretrain_dataloader.dataset, args.num_agents, args.alpha)
            active_agent_datasets = [ds for ds in agent_datasets if len(ds) > 0]
            print(f"Data split resulted in {len(active_agent_datasets)} active agents out of {args.num_agents}.")
            agent_dataloaders = [
                DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
                for ds in active_agent_datasets]

        # --- Federated Training Loop (Unchanged from your working version) ---
        for round_num in range(args.comm_rounds):
            print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
            agent_models = []
            round_agg_loss = {}
            # Iterate over the active dataloaders, not a fixed range
            for agent_dataloader in agent_dataloaders:
                agent_model = copy.deepcopy(global_model).to(device)
                loss_dict = agent_update(agent_model, agent_dataloader, args.local_epochs, criterion, device,
                                         config.LEARNING_RATE)
                if loss_dict:  # Check if the agent trained successfully
                    agent_models.append(agent_model)
                    for k, v in loss_dict.items():
                        round_agg_loss[k] = round_agg_loss.get(k, 0.0) + v
            if not agent_models:
                print(f"Round {round_num + 1} | All active agents failed to train. Stopping.")
                break
            # Average losses and models over the number of agents that actually trained
            num_successful_agents = len(agent_models)
            for k in round_agg_loss:
                round_agg_loss[k] /= num_successful_agents
            wandb.log({f"train/avg_{k}": v for k, v in round_agg_loss.items()}, step=round_num + 1)
            global_state_dict = aggregate_models(agent_models)
            global_model.load_state_dict(global_state_dict)
            if (round_num + 1) % args.eval_every == 0:
                knn_acc = knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K,
                                         config.KNN_TEMPERATURE)
                wandb.log({"eval/global_knn_accuracy": knn_acc}, step=round_num + 1)
        final_model_to_eval = global_model

    elif args.mode == 'decentralized':
        wandb.run.name = f"decentralized_{args.topology}_agents_{args.num_agents}_hetero_{args.heterogeneity_type}"
        adj_matrix = set_network_topology(args.topology, args.num_agents)
        agent_models = [copy.deepcopy(global_model).to(device) for _ in range(args.num_agents)]
        if args.heterogeneity_type == 'label_skew':
            # This logic correctly uses the pre-loaded CIFAR data with a non-IID split.
            agent_datasets = split_data(pretrain_dataloader.dataset, args.num_agents, args.alpha)
            agent_dataloaders = [
                DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS) for ds in
                agent_datasets if len(ds) > 0]
            for round_num in range(args.comm_rounds):
                print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
                for i in range(len(agent_dataloaders)):
                    agent_update(agent_models[i], agent_dataloaders[i], args.local_epochs, criterion, device,
                                 config.LEARNING_RATE)
                agent_models = gossip_average(agent_models, adj_matrix)
                if (round_num + 1) % args.eval_every == 0:
                    consensus_model = get_consensus_model(agent_models, device)
                    if consensus_model:
                        knn_acc = knn_evaluation(consensus_model, train_loader_eval, test_loader_eval, device,
                                                 config.KNN_K, config.KNN_TEMPERATURE)
                        wandb.log({"eval/consensus_knn_accuracy": knn_acc}, step=round_num + 1)
            final_model_to_eval = get_consensus_model(agent_models, device)

        elif args.heterogeneity_type == 'label_skew_personalized':
            # Path for personalized evaluation of label-skewed specialists
            agent_train_datasets, agent_test_datasets = split_train_test_data_personalized(
                train_dataset=pretrain_dataloader.dataset,
                test_dataset=test_loader_eval.dataset,
                num_agents=args.num_agents,
                non_iid_alpha=args.alpha
            )

            agent_train_dataloaders = [
                DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
                for ds in agent_train_datasets if len(ds) > 0
            ]
            agent_test_dataloaders = [
                DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
                for ds in agent_test_datasets if len(ds) > 0
            ]

            num_active_agents = len(agent_train_dataloaders)
            agent_models = agent_models[:num_active_agents]

            decentralized_personalized_training(
                agent_models=agent_models, agent_train_dataloaders=agent_train_dataloaders,
                agent_test_dataloaders=agent_test_dataloaders, adj_matrix=adj_matrix,
                criterion=criterion, device=device, comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs, eval_every=args.eval_every,
                proj_input_dim=config.PROJECTION_INPUT_DIM, eval_epochs=config.EVAL_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                global_train_loader_eval=train_loader_eval,
                global_test_loader_eval=test_loader_eval
            )
            final_model_to_eval = None

        elif args.heterogeneity_type == 'domain_shift':
            # This block correctly creates artificial domains for the CIFAR-10 dataset.
            vicreg_transform = VICRegTransform(input_size=config.INPUT_SIZE)
            domain_shifts = [
                T.Compose([]),  # Base (clean)
                T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
                T.RandomRotation(degrees=90),
                T.ColorJitter(brightness=0.5, contrast=0.5),
                T.RandomPerspective(distortion_scale=0.5, p=1.0),
            ]
            final_domain_shifts = [domain_shifts[i % len(domain_shifts)] for i in range(args.num_agents)]
            agent_specific_transforms = [
                PreTransform(pre_transform=ds, main_transform=vicreg_transform) for ds in final_domain_shifts
            ]
            # This uses the pre-loaded CIFAR dataloaders from the top of the script.
            agent_train_dataloaders, agent_test_dataloaders = get_domain_shift_dataloaders(
                train_dataset=pretrain_dataloader.dataset,
                test_dataset=test_loader_eval.dataset,
                batch_size=args.batch_size,
                num_workers=config.NUM_WORKERS,
                num_agents=args.num_agents,
                agent_transforms=agent_specific_transforms,
                dataset_config=dataset_config
            )
            decentralized_personalized_training(
                agent_models=agent_models, agent_train_dataloaders=agent_train_dataloaders,
                agent_test_dataloaders=agent_test_dataloaders, adj_matrix=adj_matrix,
                criterion=criterion, device=device, comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs, eval_every=args.eval_every,
                proj_input_dim=config.PROJECTION_INPUT_DIM, eval_epochs=config.EVAL_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                global_train_loader_eval=train_loader_eval,
                global_test_loader_eval=test_loader_eval
            )
            final_model_to_eval = None

        # Split data across agents randomly (IID classes / domains)
        elif args.heterogeneity_type == 'office_random':
            # This block now correctly uses the OfficeHome dataloaders that were prepared at the start.
            decentralized_personalized_training(
                agent_models=agent_models, agent_train_dataloaders=agent_train_dataloaders,
                agent_test_dataloaders=agent_test_dataloaders, adj_matrix=adj_matrix,
                criterion=criterion, device=device, comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs, eval_every=args.eval_every,
                proj_input_dim=config.PROJECTION_INPUT_DIM, eval_epochs=config.EVAL_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                global_train_loader_eval=train_loader_eval,
                global_test_loader_eval=test_loader_eval
            )
            final_model_to_eval = None

        # Split data across agents based on OfficeHome domains
        elif args.heterogeneity_type == 'office_domain_split':
            # This block loads data with each agent assigned a specific domain
            vicreg_transform_officehome = VICRegTransform(input_size=224)
            eval_transform_officehome = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Call the new domain-splitting function
            agent_train_dataloaders, agent_test_dataloaders = get_officehome_domain_split_loaders_personalized(
                root_dir="datasets/OfficeHomeDataset",
                num_agents=args.num_agents,
                batch_size=args.batch_size,
                num_workers=config.NUM_WORKERS,
                train_transform=vicreg_transform_officehome,
                eval_transform=eval_transform_officehome,
                num_classes=args.num_classes
            )

            decentralized_personalized_training(
                agent_models=agent_models, agent_train_dataloaders=agent_train_dataloaders,
                agent_test_dataloaders=agent_test_dataloaders, adj_matrix=adj_matrix,
                criterion=criterion, device=device, comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs, eval_every=args.eval_every,
                proj_input_dim=config.PROJECTION_INPUT_DIM, eval_epochs=config.EVAL_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                global_train_loader_eval=train_loader_eval,
                global_test_loader_eval=test_loader_eval
            )
            final_model_to_eval = None

        elif args.heterogeneity_type == 'office_hierarchical':
            # This block handles the new hierarchical experiment
            vicreg_transform_officehome = VICRegTransform(input_size=224)
            eval_transform_officehome = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            wandb.config.update({"topology": args.topology})

            # 1. Set the clustered network topology
            adj_matrix = set_hierarchical_topology(
                num_neighborhoods=args.num_neighborhoods,
                agents_per_neighborhood=args.agents_per_neighborhood
            )

            # 2. Get the hierarchical dataloaders
            agent_train_dataloaders, agent_test_dataloaders = get_officehome_hierarchical_loaders(
                root_dir="datasets/OfficeHomeDataset",
                num_neighborhoods=args.num_neighborhoods,
                agents_per_neighborhood=args.agents_per_neighborhood,
                batch_size=args.batch_size,
                num_workers=config.NUM_WORKERS,
                train_transform=vicreg_transform_officehome,
                eval_transform=eval_transform_officehome,
                num_classes=args.num_classes
            )

            # 3. Initialize models for the actual number of active agents
            num_active_agents = len(agent_train_dataloaders)
            agent_models = [copy.deepcopy(global_model).to(device) for _ in range(num_active_agents)]

            # 4. Run the personalized training and evaluation loop
            decentralized_personalized_training(
                agent_models=agent_models, agent_train_dataloaders=agent_train_dataloaders,
                agent_test_dataloaders=agent_test_dataloaders, adj_matrix=adj_matrix,
                criterion=criterion, device=device, comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs, eval_every=args.eval_every,
                proj_input_dim=config.PROJECTION_INPUT_DIM, eval_epochs=config.EVAL_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                global_train_loader_eval=train_loader_eval,
                global_test_loader_eval=test_loader_eval
            )
            if agent_models:
                evaluate_neighborhood_consensus(
                    all_agent_models=agent_models,
                    all_agent_test_dataloaders=agent_test_dataloaders,
                    num_neighborhoods=args.num_neighborhoods,
                    agents_per_neighborhood=args.agents_per_neighborhood,
                    device=device,
                    proj_input_dim=config.PROJECTION_INPUT_DIM,
                    eval_epochs=config.EVAL_EPOCHS,
                    batch_size=args.batch_size  # Pass the batch size
                )
            final_model_to_eval = None

        elif args.heterogeneity_type == 'combo_domain':
            wandb.run.name = f"combo_domain_{args.topology}_agents_{args.num_agents}"

            # This experiment is designed for the domain-shift scenario
            agent_train_dataloaders, _ = get_officehome_domain_split_loaders_personalized(
                root_dir="datasets/OfficeHomeDataset",
                num_agents=args.num_agents,
                batch_size=args.batch_size,
                num_workers=config.NUM_WORKERS,
                train_transform=vicreg_transform_officehome,
                eval_transform=eval_transform_officehome,
                num_classes=args.num_classes
            )

            public_dataloader = get_public_dataloader(
                root_dir="datasets/OfficeHomeDataset",
                batch_size=args.batch_size,
                num_workers=config.NUM_WORKERS,
                transform=vicreg_transform_officehome
            )

            # Initialize personalized backbones and a list of identical classifiers
            agent_backbones = [copy.deepcopy(global_model).to(device) for _ in range(args.num_agents)]
            agent_classifiers = [nn.Linear(config.PROJECTION_INPUT_DIM, args.num_classes).to(device) for _ in
                                 range(args.num_agents)]

            # Run the new training protocol
            final_backbones, final_classifiers = alignment_collaborative_training(
                agent_backbones=agent_backbones,
                agent_classifiers=agent_classifiers,
                agent_train_dataloaders=agent_train_dataloaders,
                global_test_loader=test_loader_eval,  # Pass the global test set for evaluation
                public_dataloader=public_dataloader,
                adj_matrix=adj_matrix,
                vicreg_criterion=criterion,
                classifier_criterion=nn.CrossEntropyLoss(),
                device=device,
                comm_rounds=args.comm_rounds,
                local_epochs=args.local_epochs,
                learning_rate=config.LEARNING_RATE,
                eval_every=args.eval_every,
                alignment_strength=args.alignment_strength
            )

            # Final evaluation: test each personalized backbone with its final shared classifier
            # print("\n--- Final Evaluation with Shared Classifier ---")
            # final_accuracies = []
            # for i in range(args.num_agents):
            #     print(f"Evaluating Agent {i}'s personalized backbone...")
            #     final_acc = evaluate_combo_model(final_backbones[i], final_classifiers[i], test_loader_eval, device)
            #     final_accuracies.append(final_acc)
            #     wandb.summary[f"agent_{i}_final_combo_accuracy"] = final_acc
            #
            # avg_acc = sum(final_accuracies) / len(final_accuracies) if final_accuracies else 0
            # wandb.summary["average_final_combo_accuracy"] = avg_acc
            # print(f"Final Average Collaborative Accuracy: {avg_acc:.2f}%")

            final_combo_evaluation(
                final_backbones=final_backbones,
                final_classifiers=final_classifiers,
                global_test_loader=test_loader_eval,
                device=device
            )

            final_model_to_eval = None

    if final_model_to_eval:
        print("\n--- Training Finished ---")
        final_linear_acc = linear_evaluation(final_model_to_eval, config.PROJECTION_INPUT_DIM, train_loader_eval,
                                             test_loader_eval, config.EVAL_EPOCHS, device)
        final_knn_acc = knn_evaluation(final_model_to_eval, train_loader_eval, test_loader_eval, device, config.KNN_K,
                                       config.KNN_TEMPERATURE)
        wandb.summary["final_linear_accuracy"] = final_linear_acc
        wandb.summary["final_knn_accuracy"] = final_knn_acc
        plot_tsne(final_model_to_eval, test_loader_eval, device,
                  plot_title=f"Final {args.mode} Model t-SNE", save_html_path=None)
        plot_pca(final_model_to_eval, test_loader_eval, device,
                 plot_title=f"Final {args.mode} Model PCA")
    wandb.finish()


if __name__ == '__main__':
    main()
