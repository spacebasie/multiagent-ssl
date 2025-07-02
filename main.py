# main.py

"""
Main script to run the VICReg self-supervised learning pipeline.

This script performs the following steps:
1. Loads configuration from the `config.py` file.
2. Initializes the VICReg model and loss function.
3. Loads the CIFAR-10 dataset.
4. Runs the self-supervised pre-training loop.
5. Saves the pre-trained backbone weights.
6. Evaluates the learned representations using linear probing and kNN.
7. Prints a final summary of the results.
"""

import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt

# Import from our modularized files
import config
from model import VICReg, VICRegLoss
from data import get_dataloaders
from evaluate import linear_evaluation, knn_evaluation


def vicreg_pretraining(model, dataloader, epochs, device, lambda_, mu, nu):
    """
    Runs the VICReg self-supervised pre-training loop.

    Args:
        model (nn.Module): The VICReg model to be trained.
        dataloader (DataLoader): DataLoader for the pre-training data.
        epochs (int): The number of epochs to train for.
        device (str): The device to run training on ('cuda' or 'cpu').
        lambda_ (float): Coefficient for the invariance term.
        mu (float): Coefficient for the variance term.
        nu (float): Coefficient for the covariance term.
    """
    criterion = VICRegLoss(lambda_=lambda_, mu=mu, nu=nu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    model.to(device)

    epoch_losses = []
    print(f"--- Starting VICReg Pre-training (λ={lambda_}, μ={mu}, ν={nu}) ---")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Unpack the two augmented views
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

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, marker='o', linestyle='-')
    plt.title(f'VICReg Training Loss (λ={lambda_}, μ={mu}, ν={nu})')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(f"vicreg_loss_l{lambda_}_m{mu}.png")
    plt.close()


def main():
    """Main function to execute the full pipeline."""
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Data Loading ---
    pretrain_loader, train_eval_loader, test_eval_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE,
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
        epochs=config.PRETRAIN_EPOCHS,
        device=device,
        lambda_=config.LAMBDA,
        mu=config.MU,
        nu=config.NU
    )

    # --- 4. Save the Pre-trained Backbone ---
    print(f"Saving backbone weights to {config.MODEL_SAVE_PATH}...")
    torch.save(model.backbone.state_dict(), config.MODEL_SAVE_PATH)
    print("Backbone weights saved successfully.")

    # --- 5. Evaluation ---
    linear_acc = linear_evaluation(
        model,
        proj_output_dim=config.PROJECTION_INPUT_DIM,
        train_loader=train_eval_loader,
        test_loader=test_eval_loader,
        epochs=config.EVAL_EPOCHS,
        device=device
    )

    knn_acc = knn_evaluation(
        model,
        train_loader=train_eval_loader,
        test_loader=test_eval_loader,
        device=device,
        k=config.KNN_K,
        temperature=config.KNN_TEMPERATURE
    )

    # --- 6. Print Final Results ---
    print("\n" + "=" * 50)
    print("           Final Performance Summary")
    print("=" * 50)
    print(f"Hyperparameters: λ={config.LAMBDA}, μ={config.MU}, ν={config.NU}")
    print(f"Pre-training Epochs: {config.PRETRAIN_EPOCHS}")
    print("-" * 50)
    print(f"Linear Evaluation Accuracy: {linear_acc:.2f}%")
    print(f"k-NN Evaluation Accuracy:   {knn_acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()