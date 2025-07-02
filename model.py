# model.py

"""
Defines the VICReg model architecture and the associated loss function.

This module contains:
- The VICReg model, which wraps a backbone (e.g., ResNet) and a projection head.
- The VICRegLoss class, which computes the variance, invariance, and covariance loss.
"""

import torch
from torch import nn
import torch.nn.functional as F
from lightly.models.modules.heads import VICRegProjectionHead

# --- 1. Model Definition ---
class VICReg(nn.Module):
    """
    VICReg model combining a backbone and a projection head.

    Args:
        backbone (nn.Module): The feature extractor network (e.g., a ResNet).
        proj_input_dim (int): The output dimension of the backbone.
        proj_hidden_dim (int): The hidden dimension of the projection head.
        proj_output_dim (int): The final output dimension of the projection head.
    """
    def __init__(self, backbone: nn.Module, proj_input_dim: int, proj_hidden_dim: int, proj_output_dim: int):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the backbone and projection head.
        This is used during the pre-training phase.

        Args:
            x (torch.Tensor): The input tensor (batch of images).

        Returns:
            torch.Tensor: The final projected representations.
        """
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through only the backbone.
        This is used during the evaluation phase (linear probing, kNN).

        Args:
            x (torch.Tensor): The input tensor (batch of images).

        Returns:
            torch.Tensor: The flattened backbone representations.
        """
        x = self.backbone(x)
        return x.flatten(start_dim=1)

# --- 2. VICReg Loss Definition ---
class VICRegLoss(nn.Module):
    """
    VICReg Loss Function.

    Args:
        lambda_ (float): Coefficient for the invariance term (sim_loss).
        mu (float): Coefficient for the variance term (std_loss).
        nu (float): Coefficient for the covariance term (cov_loss).
        epsilon (float): Small value for numerical stability in variance calculation.
    """
    def __init__(self, lambda_: float = 25.0, mu: float = 25.0, nu: float = 1.0, epsilon: float = 1e-4):
        super().__init__()
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Computes the VICReg loss from two batches of projected representations.

        Args:
            z_a (torch.Tensor): Representations from the first augmented view.
            z_b (torch.Tensor): Representations from the second augmented view.

        Returns:
            torch.Tensor: The final combined VICReg loss.
        """
        # Invariance term: Encourages representations of two views to be similar.
        sim_loss = F.mse_loss(z_a, z_b)

        # Variance term: Encourages the variance along each dimension to be close to 1.
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # Covariance term: Decorrelates the dimensions of the representations.
        z_a_norm = z_a - z_a.mean(dim=0)
        z_b_norm = z_b - z_b.mean(dim=0)
        N, D = z_a.shape
        cov_z_a = (z_a_norm.T @ z_a_norm) / (N - 1)
        cov_z_b = (z_b_norm.T @ z_b_norm) / (N - 1)

        # Zero out the diagonal elements to only penalize off-diagonal covariance
        off_diag_mask = ~torch.eye(D, device=z_a.device, dtype=torch.bool)
        cov_loss = (cov_z_a[off_diag_mask].pow_(2).sum() / D) + \
                   (cov_z_b[off_diag_mask].pow_(2).sum() / D)

        # Combine the three terms with their respective coefficients
        loss = (self.lambda_ * sim_loss) + (self.mu * std_loss) + (self.nu * cov_loss)
        return loss