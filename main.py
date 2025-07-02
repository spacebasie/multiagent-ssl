# main.py

"""
Main script to run the VICReg self-supervised learning pipeline.

This script can be configured via command-line arguments.
Run `python main.py --help` to see all available options.
"""

import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import argparse  # Import the argparse library

# Import from our modularized files
import config
from model import VICReg, VICRegLoss
from data import get_dataloaders
from evaluate import linear_evaluation, knn_evaluation


def parse_arguments():
    """
    Parses command-line arguments.

    Uses values from config.py as defaults, allowing them to be
    overridden from the command line.
    """
    parser = argparse.ArgumentParser(description="VICReg Training and Evaluation Pipeline")

    parser.add_argument('--device', type=str, default=config.DEVICE,
                        help='Device to use for training (e.g., "cuda" or "cpu")')

    # --- Pre-training Arguments ---
    parser.add_argument('--pretrain_epochs', type=int, default=config.PRETRAIN_EPOCHS,
                        help='Number of epochs for self-supervised pre-training.')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training and evaluation.')

    # --- VICReg Loss Arguments ---
    parser.add_argument('--lambda_val', type=float, default=config.LAMBDA,
                        help='Lambda coefficient for the VICReg invariance term.')
    parser.add_argument('--mu_val', type=float, default=config.MU,
                        help='Mu coefficient for the VICReg variance term.')

    # --- Evaluation Arguments ---
    parser.add_argument('--eval_epochs', type=int, default=config.EVAL_EPOCHS,
                        help='Number of epochs for linear evaluation.')
    parser.add_argument('--knn_k', type=int, default=config.KNN_K,
                        help='Number of neighbors for kNN evaluation.')

    return parser.parse_args()


def vicreg_pretraining(model, dataloader, epochs, device, lambda_, mu, nu):
    # This function remains the same as before
    criterion = VICRegLoss(lambda_=lambda_, mu=mu, nu=nu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    model.to(device)

    epoch_losses = []
    print(f"--- Starting VICReg Pre-training (λ={lambda_}, μ={mu}, ν={nu}) ---")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x0, x1 = batch[0]
            x0, x1 = x0.to(device), x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            if torch.isnan(loss):
                print(f"Stopping training due to NaN loss at epoch {epoch}.")
                return
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss.cpu().item())
        print(f"Pre-training Epoch: {epoch:02}, Loss: {avg_loss:.5f}")
    print("--- Pre-training Finished ---")

    # Plotting code remains the same...


def main():
    """Main function to execute the full pipeline."""
    args = parse_arguments()  # Parse arguments at the beginning

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Data Loading ---
    # Use parsed arguments instead of config values directly
    pretrain_loader, train_eval_loader, test_eval_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        input_size=config.INPUT_SIZE,
        path=config.DATASET_PATH
    )

    # --- 2. Model Initialization ---
    backbone = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
    model = VICReg(
        backbone,
        proj_input_dim=config.PROJECTION_INPUT_DIM,
        proj_hidden_dim=config.PROJECTION_HIDDEN_DIM,
        proj_output_dim=config.PROJECTION_OUTPUT_DIM
    ).to(device)

    # --- 3. Pre-training ---
    vicreg_pretraining(
        model,
        pretrain_loader,
        epochs=args.pretrain_epochs,
        device=device,
        lambda_=args.lambda_val,
        mu=args.mu_val,
        nu=config.NU  # nu is kept fixed from config
    )

    # --- 4. Save the Pre-trained Backbone ---
    # (Code remains the same)

    # --- 5. Evaluation ---
    linear_acc = linear_evaluation(
        model,
        proj_output_dim=config.PROJECTION_INPUT_DIM,
        train_loader=train_eval_loader,
        test_loader=test_eval_loader,
        epochs=args.eval_epochs,
        device=device
    )

    knn_acc = knn_evaluation(
        model,
        train_loader=train_eval_loader,
        test_loader=test_eval_loader,
        device=device,
        k=args.knn_k,
        temperature=config.KNN_TEMPERATURE
    )

    # --- 6. Print Final Results ---
    print("\n" + "=" * 50)
    print("           Final Performance Summary")
    print("=" * 50)
    print(f"Hyperparameters: λ={args.lambda_val}, μ={args.mu_val}, ν={config.NU}")
    print(f"Pre-training Epochs: {args.pretrain_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("-" * 50)
    print(f"Linear Evaluation Accuracy: {linear_acc:.2f}%")
    print(f"k-NN Evaluation Accuracy:   {knn_acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()