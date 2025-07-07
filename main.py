# main.py

"""
Main script to run the VICReg self-supervised learning pipeline.

This script simulates a decentralized/federated training environment,
generates a Markdown report of the results, and optionally saves the model.
To run a traditional centralized training, set `--num_agents 1`.

Run `python main.py --help` to see all available options.
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import copy
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import from our modularized files
import config
from model import VICReg, VICRegLoss
from data_splitter import split_data
from evaluate import linear_evaluation, knn_evaluation
from lightly.transforms.vicreg_transform import VICRegTransform


# --- Helper Functions (agent_update, aggregate_models) remain the same ---

def agent_update(agent_model, agent_dataloader, local_epochs, device):
    """Performs the local training for a single agent."""
    criterion = VICRegLoss(lambda_=config.LAMBDA, mu=config.MU, nu=config.NU)
    optimizer = torch.optim.SGD(agent_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    agent_model.train()
    for _ in range(local_epochs):
        for batch in agent_dataloader:
            x0, x1 = batch[0]
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = agent_model(x0), agent_model(x1)
            loss = criterion(z0, z1)
            if torch.isnan(loss):
                print(f"  Warning: NaN loss detected for an agent. Skipping update.")
                return
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def aggregate_models(agent_models):
    """Averages the weights of the agent models."""
    global_state_dict = copy.deepcopy(agent_models[0].state_dict())
    for key in global_state_dict.keys():
        for i in range(1, len(agent_models)):
            global_state_dict[key] += agent_models[i].state_dict()[key]
        global_state_dict[key] = torch.div(global_state_dict[key], len(agent_models))
    return global_state_dict


# --- Plotting and Reporting Functions ---

def plot_learning_curve(eval_points, accuracies, filename="learning_curve.png"):
    """Plots and saves the learning curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(eval_points, accuracies, marker='o', linestyle='-')
    plt.title("kNN Accuracy vs. Communication Round")
    plt.xlabel("Communication Round")
    plt.ylabel("kNN Accuracy (%)")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")


def generate_markdown_report(args, linear_acc, knn_acc, plot_filename, report_filename):
    """Generates a Markdown report summarizing the experiment."""
    with open(report_filename, 'w') as f:
        # Title and Date
        f.write("# VICReg Experiment Report\n\n")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"**Report generated on:** {timestamp}\n\n")

        # Parameters Table
        f.write("## Experiment Parameters\n\n")
        f.write("| Parameter                | Value |\n")
        f.write("|--------------------------|-------|\n")
        f.write(f"| Number of Agents         | {args.num_agents} |\n")
        f.write(f"| Communication Rounds     | {args.comm_rounds} |\n")
        f.write(f"| Local Epochs per Round   | {args.local_epochs} |\n")
        alpha_value_str = f"{args.alpha:.2f}" if args.num_agents > 1 else "N/A"
        f.write(f"| Non-IID Alpha            | {alpha_value_str} |\n")
        f.write(f"| Batch Size               | {config.BATCH_SIZE} |\n")
        f.write(f"| VICReg Lambda            | {config.LAMBDA} |\n")
        f.write(f"| VICReg Mu                | {config.MU} |\n\n")

        # Final Results
        f.write("## Final Evaluation Results\n\n")
        f.write(f"- **Final Linear Evaluation Accuracy:** {linear_acc:.2f}%\n")
        f.write(f"- **Final k-NN Evaluation Accuracy:** {knn_acc:.2f}%\n\n")

        # Learning Curve Plot
        f.write("## Learning Curve\n\n")
        # The image is linked using a relative path, so it will render correctly
        # when viewed on GitHub or with a local Markdown viewer.
        f.write(f"![kNN Accuracy vs. Communication Round]({plot_filename})\n")

    print(f"Markdown report saved to {report_filename}")


# --- Main Training Logic ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Decentralized VICReg Training Pipeline")

    parser.add_argument('--num_agents', type=int, default=config.NUM_AGENTS)
    parser.add_argument('--comm_rounds', type=int, default=config.COMMUNICATION_ROUNDS)
    parser.add_argument('--local_epochs', type=int, default=config.LOCAL_EPOCHS)
    parser.add_argument('--alpha', type=float, default=config.NON_IID_ALPHA)
    parser.add_argument('--eval_epochs', type=int, default=config.EVAL_EPOCHS)
    parser.add_argument('--eval_every', type=int, default=5)

    # Argument to control saving weights
    parser.add_argument('--save_weights', action='store_true',
                        help='If set, saves the final model backbone weights.')

    return parser.parse_args()


def main():
    """Main function to execute the full pipeline."""
    args = parse_arguments()
    device = config.DEVICE if torch.cuda.is_available() else "cpu"

    print("--- Starting VICReg Experiment ---")
    print(f"Device: {device}, Agents: {args.num_agents}, Rounds: {args.comm_rounds}, Local Epochs: {args.local_epochs}")
    if args.num_agents > 1: print(f"Non-IID Alpha: {args.alpha}")

    # --- Data Preparation ---
    transform_vicreg = VICRegTransform(input_size=config.INPUT_SIZE)
    full_train_dataset = torchvision.datasets.CIFAR10(root=config.DATASET_PATH, download=True, train=True,
                                                      transform=transform_vicreg)
    agent_datasets = split_data(full_train_dataset, args.num_agents, args.alpha)
    agent_dataloaders = [DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) for
                         ds in agent_datasets]
    transform_eval = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                      (0.2023, 0.1994, 0.2010))])
    train_dataset_eval = torchvision.datasets.CIFAR10(root=config.DATASET_PATH, train=True, transform=transform_eval)
    test_dataset_eval = torchvision.datasets.CIFAR10(root=config.DATASET_PATH, train=False, transform=transform_eval)
    train_loader_eval = DataLoader(train_dataset_eval, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader_eval = DataLoader(test_dataset_eval, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
    global_model = VICReg(backbone, proj_input_dim=config.PROJECTION_INPUT_DIM,
                          proj_hidden_dim=config.PROJECTION_HIDDEN_DIM,
                          proj_output_dim=config.PROJECTION_OUTPUT_DIM).to(device)

    # --- Training Loop ---
    eval_points = []
    accuracies = []
    for round_num in range(args.comm_rounds):
        print(f"\n--- Round {round_num + 1}/{args.comm_rounds} ---")
        agent_models = []
        for i in range(args.num_agents):
            agent_model = copy.deepcopy(global_model).to(device)
            agent_update(agent_model, agent_dataloaders[i], args.local_epochs, device)
            agent_models.append(agent_model)

        global_state_dict = aggregate_models(agent_models)
        global_model.load_state_dict(global_state_dict)

        if (round_num + 1) % args.eval_every == 0:
            print(f"\n--- Evaluating at Round {round_num + 1} ---")
            knn_acc = knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K,
                                     config.KNN_TEMPERATURE)
            eval_points.append(round_num + 1)
            accuracies.append(knn_acc)

    # --- Final Actions ---
    print("\n--- Training Finished ---")

    # Generate unique filenames for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"run_{run_timestamp}_learning_curve.png"
    report_filename = f"run_{run_timestamp}_report.md"  # Changed to .md

    print(f"\n--- Plotting Results ---")
    if eval_points:
        print(f"\n--- Saving plot ---")
        plot_learning_curve(eval_points, accuracies, filename=plot_filename)

    # Run final, detailed evaluations
    print("\n--- Starting Final Evaluation ---")
    final_linear_acc = linear_evaluation(global_model, config.PROJECTION_INPUT_DIM, train_loader_eval, test_loader_eval,
                                         config.EVAL_EPOCHS, device)
    final_knn_acc = knn_evaluation(global_model, train_loader_eval, test_loader_eval, device, config.KNN_K,
                                   config.KNN_TEMPERATURE)

    # Generate the Markdown report
    generate_markdown_report(args, final_linear_acc, final_knn_acc, plot_filename, report_filename)

    # Optionally save the model weights
    if args.save_weights:
        model_save_path = f"run_{run_timestamp}_{config.MODEL_SAVE_PATH}"
        print(f"\n--- Saving final model weights to {model_save_path} ---")
        torch.save(global_model.backbone.state_dict(), model_save_path)
    else:
        print("\n--- Skipping model weight saving as per configuration ---")


if __name__ == '__main__':
    main()
