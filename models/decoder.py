"""
Preset Decoder for the SpinVAE-style FM synthesizer VAE.

Maps a latent vector z ∈ R^Lz back to a 6-dimensional preset reconstruction
û ∈ R^6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PresetDecoder(nn.Module):
    """Decode a latent vector into FM preset parameters.

    Architecture::

        z (latent_dim,) → Linear(latent_dim→hidden) → ReLU
                        → Linear(hidden→output_dim)  [û]

    Parameters
    ----------
    latent_dim : dimensionality of the latent space (default 16).
    hidden_dim : width of the hidden layer (default 32).
    output_dim : dimensionality of the preset vector (default 6).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 6,
    ) -> None:
        super().__init__()
        self.fc1    = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent *z* into a preset reconstruction.

        Parameters
        ----------
        z : Tensor of shape (B, latent_dim).

        Returns
        -------
        u_hat : Tensor (B, output_dim).
        """
        x     = F.relu(self.fc1(z))
        u_hat = self.fc_out(x)
        return u_hat
