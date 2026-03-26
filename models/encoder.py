"""
Preset Encoder for the SpinVAE-style FM synthesizer VAE.

Maps a 6-dimensional preset vector u ∈ R^6 to the parameters of a diagonal
Gaussian in latent space: μ ∈ R^Lz and log σ² ∈ R^Lz.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PresetEncoder(nn.Module):
    """Encode FM preset parameters into a Gaussian latent distribution.

    Architecture::

        u (6,) → Linear(6→hidden) → ReLU → Linear(hidden→latent_dim)  [μ]
                                          → Linear(hidden→latent_dim)  [log σ²]

    Parameters
    ----------
    input_dim  : dimensionality of the preset vector (default 6).
    hidden_dim : width of the hidden layer (default 32).
    latent_dim : dimensionality of the latent space (default 16).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.fc1      = nn.Linear(input_dim,  hidden_dim)
        self.fc_mu    = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, u: torch.Tensor):
        """Encode preset *u* into (μ, log σ²).

        Parameters
        ----------
        u : Tensor of shape (B, input_dim).

        Returns
        -------
        mu     : Tensor (B, latent_dim) – mean of the posterior.
        logvar : Tensor (B, latent_dim) – log-variance of the posterior.
        """
        x      = F.relu(self.fc1(u))
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
