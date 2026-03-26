"""
SpinVAE-style training script for the FM Preset VAE.

Training schedule
-----------------
Phase 2 : Train VAE with L_preset + β · L_DKL.
Phase 3 : Fine-tune with timbre regularisation:
          L_preset + β · L_DKL + γ · L_timbre (AR-loss).

Usage::

    python scripts/train_model.py [--config CONFIG] [--phase {2,3,all}]
                                  [--checkpoint CKPT] [--device DEVICE]

Examples::

    # Train both phases with the default config
    python scripts/train_model.py

    # Resume from a checkpoint for Phase 3 only
    python scripts/train_model.py --phase 3 --checkpoint checkpoints/phase2_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Allow running from the repository root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore[assignment]

from dataset.dataset import FMPresetDataset
from models.vae import PresetVAE, loss_vae, ar_loss

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load a YAML config file, falling back to hard-coded defaults."""
    if yaml is not None and os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f)
    # Minimal defaults so the script works without PyYAML
    return {
        "data": {
            "metadata_csv": "data/metadata.csv",
            "splits_dir":   "data/splits",
            "audio_dir":    "data/audio",
        },
        "model":    {"input_dim": 6, "hidden_dim": 32, "latent_dim": 16},
        "training": {
            "seed": 42, "batch_size": 256, "num_workers": 0,
            "phase2_epochs": 100, "phase2_lr": 1e-3, "beta": 1.0,
            "phase3_epochs": 50,  "phase3_lr": 1e-4,
            "gamma": 0.1, "ar_delta": 1.0,
            "compute_audio_features": False,
        },
        "output": {"checkpoint_dir": "checkpoints", "log_interval": 10},
    }


def _nested_get(cfg: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ---------------------------------------------------------------------------
# Dataset / DataLoader factory
# ---------------------------------------------------------------------------

def _make_loader(
    cfg:      dict,
    split:    str,
    audio:    bool,
    shuffle:  bool = True,
) -> DataLoader:
    data_cfg     = cfg["data"]
    train_cfg    = cfg["training"]
    splits_dir   = data_cfg["splits_dir"]
    split_file   = os.path.join(splits_dir, f"{split}.npy")

    dataset = FMPresetDataset(
        metadata_csv           = data_cfg["metadata_csv"],
        split_file             = split_file if os.path.isfile(split_file) else None,
        audio_dir              = data_cfg.get("audio_dir"),
        compute_audio_features = audio,
    )
    loader = DataLoader(
        dataset,
        batch_size  = int(train_cfg["batch_size"]),
        shuffle     = shuffle,
        num_workers = int(train_cfg.get("num_workers", 0)),
        pin_memory  = torch.cuda.is_available(),
    )
    return loader


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _train_phase2(
    model:     PresetVAE,
    cfg:       dict,
    device:    torch.device,
    ckpt_dir:  str,
) -> None:
    """Phase 2: VAE training (L_preset + β · L_DKL)."""
    train_cfg = cfg["training"]
    beta      = float(train_cfg.get("beta", 1.0))
    epochs    = int(train_cfg.get("phase2_epochs", 100))
    lr        = float(train_cfg.get("phase2_lr", 1e-3))
    log_every = int(_nested_get(cfg, "output", "log_interval", default=10))

    loader_train = _make_loader(cfg, "train", audio=False, shuffle=True)
    loader_val   = _make_loader(cfg, "val",   audio=False, shuffle=False)

    optimiser = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    print(f"\n{'='*60}")
    print(f"  Phase 2 – VAE training  ({epochs} epochs, β={beta})")
    print(f"  Train batches: {len(loader_train)}, Val batches: {len(loader_val)}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # ---- Training ---------------------------------------------------
        model.train()
        total_loss = total_preset = total_dkl = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader_train, 1):
            u = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)

            optimiser.zero_grad()
            u_hat, mu, logvar = model(u)
            loss, lp, ld      = loss_vae(u, u_hat, mu, logvar, beta=beta)
            loss.backward()
            optimiser.step()

            total_loss   += loss.item()
            total_preset += lp.item()
            total_dkl    += ld.item()
            n_batches    += 1

            if batch_idx % log_every == 0:
                print(
                    f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader_train)} "
                    f"| loss={loss.item():.4f}  preset={lp.item():.4f}  KL={ld.item():.4f}"
                )

        # ---- Validation -------------------------------------------------
        model.eval()
        val_loss = val_preset = val_dkl = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in loader_val:
                u = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
                u_hat, mu, logvar = model(u)
                loss, lp, ld      = loss_vae(u, u_hat, mu, logvar, beta=beta)
                val_loss   += loss.item()
                val_preset += lp.item()
                val_dkl    += ld.item()
                n_val      += 1

        print(
            f"  Epoch {epoch:3d} ║ "
            f"train loss={total_loss/n_batches:.4f} "
            f"| val loss={val_loss/max(n_val,1):.4f} "
            f"(preset={val_preset/max(n_val,1):.4f}, "
            f"KL={val_dkl/max(n_val,1):.4f})"
        )

    # ---- Save checkpoint ------------------------------------------------
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "phase2_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")


def _train_phase3(
    model:     PresetVAE,
    cfg:       dict,
    device:    torch.device,
    ckpt_dir:  str,
) -> None:
    """Phase 3: Fine-tune with timbre regularisation (AR-loss)."""
    train_cfg = cfg["training"]
    beta      = float(train_cfg.get("beta",    1.0))
    gamma     = float(train_cfg.get("gamma",   0.1))
    delta     = float(train_cfg.get("ar_delta", 1.0))
    epochs    = int(train_cfg.get("phase3_epochs", 50))
    lr        = float(train_cfg.get("phase3_lr", 1e-4))
    log_every = int(_nested_get(cfg, "output", "log_interval", default=10))

    loader_train = _make_loader(cfg, "train", audio=True, shuffle=True)
    loader_val   = _make_loader(cfg, "val",   audio=True, shuffle=False)

    optimiser = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    print(f"\n{'='*60}")
    print(f"  Phase 3 – Timbre regularisation  ({epochs} epochs, β={beta}, γ={gamma})")
    print(f"  Train batches: {len(loader_train)}, Val batches: {len(loader_val)}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # ---- Training ---------------------------------------------------
        model.train()
        total_loss = total_preset = total_dkl = total_ar = 0.0
        n_batches  = 0

        for batch_idx, (u, a) in enumerate(loader_train, 1):
            u = u.to(device)
            a = a.to(device)

            optimiser.zero_grad()
            u_hat, mu, logvar = model(u)
            vae_loss, lp, ld  = loss_vae(u, u_hat, mu, logvar, beta=beta)
            ar                = ar_loss(mu, a, delta=delta)
            loss              = vae_loss + gamma * ar
            loss.backward()
            optimiser.step()

            total_loss   += loss.item()
            total_preset += lp.item()
            total_dkl    += ld.item()
            total_ar     += ar.item()
            n_batches    += 1

            if batch_idx % log_every == 0:
                print(
                    f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader_train)} "
                    f"| loss={loss.item():.4f}  preset={lp.item():.4f}"
                    f"  KL={ld.item():.4f}  AR={ar.item():.4f}"
                )

        # ---- Validation -------------------------------------------------
        model.eval()
        val_loss = val_preset = val_dkl = val_ar = 0.0
        n_val = 0
        with torch.no_grad():
            for u, a in loader_val:
                u = u.to(device)
                a = a.to(device)
                u_hat, mu, logvar = model(u)
                vae_loss, lp, ld  = loss_vae(u, u_hat, mu, logvar, beta=beta)
                ar                = ar_loss(mu, a, delta=delta)
                loss              = vae_loss + gamma * ar
                val_loss   += loss.item()
                val_preset += lp.item()
                val_dkl    += ld.item()
                val_ar     += ar.item()
                n_val      += 1

        print(
            f"  Epoch {epoch:3d} ║ "
            f"train loss={total_loss/n_batches:.4f} "
            f"| val loss={val_loss/max(n_val,1):.4f} "
            f"(preset={val_preset/max(n_val,1):.4f}, "
            f"KL={val_dkl/max(n_val,1):.4f}, "
            f"AR={val_ar/max(n_val,1):.4f})"
        )

    # ---- Save checkpoint ------------------------------------------------
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "phase3_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SpinVAE FM Preset VAE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--phase", type=str, default="all", choices=["2", "3", "all"],
        help="Which training phase(s) to run.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a model checkpoint to load before training.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device (e.g. 'cpu', 'cuda', 'mps'). "
             "Auto-detected if not specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = _load_config(args.config)

    # ---- Reproducibility ------------------------------------------------
    seed = int(_nested_get(cfg, "training", "seed", default=42))
    torch.manual_seed(seed)

    # ---- Device ---------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Model ----------------------------------------------------------
    model_cfg = cfg["model"]
    model = PresetVAE(
        input_dim  = int(model_cfg.get("input_dim",  6)),
        hidden_dim = int(model_cfg.get("hidden_dim", 32)),
        latent_dim = int(model_cfg.get("latent_dim", 16)),
    )

    if args.checkpoint and os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")

    ckpt_dir = _nested_get(cfg, "output", "checkpoint_dir", default="checkpoints")

    # ---- Run requested phase(s) ----------------------------------------
    run_phase2 = args.phase in ("2", "all")
    run_phase3 = args.phase in ("3", "all")

    if run_phase2:
        _train_phase2(model, cfg, device, ckpt_dir)

    if run_phase3:
        compute_audio = bool(_nested_get(cfg, "training", "compute_audio_features", default=False))
        if not compute_audio:
            print(
                "\n[WARNING] Phase 3 requires audio features (compute_audio_features: true "
                "in config).  Skipping Phase 3."
            )
        else:
            _train_phase3(model, cfg, device, ckpt_dir)


if __name__ == "__main__":
    main()
