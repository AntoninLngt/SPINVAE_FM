"""
SpinVAE-style Preset VAE for the FM synthesizer.

Components
----------
* PresetVAE  – full encoder-reparameterize-decoder pipeline.
* loss_vae   – L_preset (MSE) + β · L_DKL.
* ar_loss    – timbre alignment regularization (AR-loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import PresetEncoder
from models.decoder import PresetDecoder


# ---------------------------------------------------------------------------
# Main VAE model
# ---------------------------------------------------------------------------

class PresetVAE(nn.Module):
    """Variational Auto-Encoder for FM synthesizer presets.

    Encoder : u ∈ R^input_dim → (μ, log σ²) ∈ R^latent_dim × R^latent_dim
    Decoder : z ∈ R^latent_dim → û ∈ R^input_dim

    Parameters
    ----------
    input_dim  : preset dimensionality (default 6).
    hidden_dim : hidden layer width shared by encoder and decoder (default 32).
    latent_dim : latent space dimensionality (default 16).
    """

    def __init__(
        self,
        input_dim:  int = 6,
        hidden_dim: int = 32,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = PresetEncoder(input_dim,  hidden_dim, latent_dim)
        self.decoder = PresetDecoder(latent_dim, hidden_dim, input_dim)

    # ------------------------------------------------------------------
    # Reparameterization
    # ------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + σ · ε  with  ε ~ N(0, I).

        During evaluation the mean is returned deterministically.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, u: torch.Tensor):
        """Encode, sample, and decode a preset batch.

        Parameters
        ----------
        u : Tensor (B, input_dim) – normalised preset parameters.

        Returns
        -------
        u_hat  : Tensor (B, input_dim) – reconstructed presets.
        mu     : Tensor (B, latent_dim) – posterior mean.
        logvar : Tensor (B, latent_dim) – posterior log-variance.
        """
        mu, logvar = self.encoder(u)
        z          = self.reparameterize(mu, logvar)
        u_hat      = self.decoder(z)
        return u_hat, mu, logvar

    # ------------------------------------------------------------------
    # Convenience: encode / decode individually
    # ------------------------------------------------------------------

    def encode(self, u: torch.Tensor):
        """Return (μ, log σ²) for preset batch *u*."""
        return self.encoder(u)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstructed presets for latent batch *z*."""
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_vae(
    u:      torch.Tensor,
    u_hat:  torch.Tensor,
    mu:     torch.Tensor,
    logvar: torch.Tensor,
    beta:   float = 1.0,
):
    """SpinVAE-style VAE loss: L_preset + β · L_DKL.

    Parameters
    ----------
    u      : ground-truth presets (B, D).
    u_hat  : reconstructed presets (B, D).
    mu     : posterior mean (B, Lz).
    logvar : posterior log-variance (B, Lz).
    beta   : weight on the KL divergence term.

    Returns
    -------
    total   : scalar total loss.
    L_preset: scalar MSE reconstruction loss.
    L_dkl   : scalar KL divergence.
    """
    L_preset = F.mse_loss(u_hat, u, reduction="mean")
    # KL(q(z|x) || N(0,I)) = -0.5 · Σ (1 + log σ² - μ² - σ²)
    L_dkl    = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total    = L_preset + beta * L_dkl
    return total, L_preset, L_dkl


def ar_loss(
    mu:    torch.Tensor,
    a:     torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Timbre alignment regularization (AR-loss).

    Encourages monotonic alignment between latent dimensions and
    normalised audio features by penalising sign disagreements in
    pairwise differences.

    Parameters
    ----------
    mu    : posterior means (B, Lz).
    a     : normalised audio features (B, F).
    delta : steepness of the tanh approximation (default 1.0).

    Returns
    -------
    loss : scalar AR loss (averaged over feature dimensions).
    """
    # Pairwise differences (B, B, *)
    D_mu = mu.unsqueeze(0) - mu.unsqueeze(1)   # (B, B, Lz)
    D_a  = a.unsqueeze(0)  - a.unsqueeze(1)    # (B, B, F)

    # Align the first min(F, Lz) audio features with the corresponding latent dims.
    n_features = min(a.shape[1], mu.shape[1])
    loss = torch.tensor(0.0, device=mu.device)
    for j in range(n_features):
        d_mu_j = D_mu[:, :, j]
        d_a_j  = D_a[:, :, j]
        loss = loss + F.l1_loss(
            torch.tanh(delta * d_mu_j),
            torch.sign(d_a_j),
        )
    return loss / n_features if n_features > 0 else loss
