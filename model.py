# model.py

"""
Defines the VICReg model architecture and the associated loss function.
The loss function now returns a detailed breakdown of its components.
"""

import torch
from torch import nn
import torch.nn.functional as F
from lightly.models.modules.heads import VICRegProjectionHead

# --- 1. Model Definition ---
class VICReg(nn.Module):
    """
    VICReg model combining a backbone and a projection head.
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
        """Performs a forward pass through the backbone and projection head."""
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through only the backbone."""
        x = self.backbone(x)
        return x.flatten(start_dim=1)

# --- 2. VICReg Loss Definition ---
class VICRegLoss(nn.Module):
    """
    VICReg Loss Function. Now returns a dictionary with detailed loss components.
    """
    def __init__(self, lambda_: float = 25.0, mu: float = 25.0, nu: float = 1.0, epsilon: float = 1e-4):
        super().__init__()
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
        """
        Computes the VICReg loss and returns a dictionary of its components.
        """
        # Invariance term
        sim_loss = F.mse_loss(z_a, z_b)

        # Variance term
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # Covariance term
        z_a_norm = z_a - z_a.mean(dim=0)
        z_b_norm = z_b - z_b.mean(dim=0)
        N, D = z_a.shape
        cov_z_a = (z_a_norm.T @ z_a_norm) / (N - 1)
        cov_z_b = (z_b_norm.T @ z_b_norm) / (N - 1)
        off_diag_mask = ~torch.eye(D, device=z_a.device, dtype=torch.bool)
        cov_loss = (cov_z_a[off_diag_mask].pow_(2).sum() / D) + \
                   (cov_z_b[off_diag_mask].pow_(2).sum() / D)

        # Combine the terms
        total_loss = (self.lambda_ * sim_loss) + (self.mu * std_loss) + (self.nu * cov_loss)

        # Return a dictionary for detailed logging
        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "variance_loss": std_loss,
            "covariance_loss": cov_loss,
            "output_std": std_z_a.mean() # Diagnostic for representation collapse
        }
