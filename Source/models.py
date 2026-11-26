"""
models.py
---------
GPU-efficient single-cell batch integration and three-way cell-type classification
tuned for a 4 GB VRAM GPU (e.g., NVIDIA RTX 2050 Laptop GPU).

Models implemented:
  1. scVI (Deep Generative VAE): Batch-corrected 10-D latent representation + Leiden clustering.
  2. CellTypist (Pretrained Reference): Zero-shot immune atlas mapping with majority voting.
  3. scANVI (Semi-Supervised VAE): Fine-tuned from scVI with 50% masked labels.
  4. LatentMLP (Supervised PyTorch Classifier): Compact MLP trained on X_scVI embeddings.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import celltypist
from celltypist import models as ct_models
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Harmonization map from fine-grained CellTypist Immune_All_Low labels to PBMC lineages
CELLTYPIST_TO_PBMC_MAP: dict[str, str] = {
    "Tcm/Naive helper T cells": "CD4 T cells",
    "Tem/Effector helper T cells": "CD4 T cells",
    "Tem/Trm helper T cells": "CD4 T cells",
    "Regulatory T cells": "CD4 T cells",
    "T follicular helper cells": "CD4 T cells",
    "Type 1 helper T cells": "CD4 T cells",
    "Type 17 helper T cells": "CD4 T cells",
    "Tcm/Naive cytotoxic T cells": "CD8 T cells",
    "Tem/Effector cytotoxic T cells": "CD8 T cells",
    "Tem/Trm cytotoxic T cells": "CD8 T cells",
    "MAIT cells": "CD8 T cells",
    "Gamma-delta T cells": "CD8 T cells",
    "Naive B cells": "B cells",
    "Memory B cells": "B cells",
    "Age-associated B cells": "B cells",
    "Germinal center B cells": "B cells",
    "Plasma cells": "B cells",
    "Plasmablasts": "B cells",
    "Classical monocytes": "CD14+ Monocytes",
    "Intermediate macrophages": "CD14+ Monocytes",
    "Non-classical monocytes": "FCGR3A+ Monocytes",
    "CD16+ NK cells": "NK cells",
    "CD16- NK cells": "NK cells",
    "ILC": "NK cells",
    "ILC3": "NK cells",
    "DC": "Dendritic Cells",
    "DC1": "Dendritic Cells",
    "DC2": "Dendritic Cells",
    "Migratory DCs": "Dendritic Cells",
    "pDC": "Dendritic Cells",
    "Megakaryocytes/platelets": "Megakaryocytes",
}


def harmonize_celltypist_labels(raw_preds: list[str] | np.ndarray) -> list[str]:
    """Map fine-grained CellTypist predictions onto canonical PBMC ontology labels."""
    harmonized = []
    for label in raw_preds:
        if label in CELLTYPIST_TO_PBMC_MAP:
            harmonized.append(CELLTYPIST_TO_PBMC_MAP[label])
        elif "helper T" in label or "CD4" in label:
            harmonized.append("CD4 T cells")
        elif "cytotoxic T" in label or "CD8" in label:
            harmonized.append("CD8 T cells")
        elif "B cell" in label or "Plasma" in label:
            harmonized.append("B cells")
        elif "NK" in label:
            harmonized.append("NK cells")
        elif "DC" in label or "Dendritic" in label:
            harmonized.append("Dendritic Cells")
        elif "Megakaryocyte" in label or "platelet" in label.lower():
            harmonized.append("Megakaryocytes")
        elif "Non-classical" in label:
            harmonized.append("FCGR3A+ Monocytes")
        elif "monocyte" in label.lower():
            harmonized.append("CD14+ Monocytes")
        else:
            harmonized.append("Other")
    return harmonized


def reset_vram_stats() -> None:
    """Clear CUDA cache and reset peak memory tracking between stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_vram_mb() -> float:
    """Return peak GPU memory allocated in MB since last reset."""
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / (1024**2), 2)


class LatentMLP(nn.Module):
    """Lightweight 2-layer MLP classifier trained on 10-D scVI latent representations."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_scvi_integration(
    adata: sc.AnnData,
    artifact_dir: Path,
    max_epochs: int = 100,
    batch_size: int = 128,
    leiden_resolution: float = 0.5,
    retrain: bool = False,
) -> tuple[scvi.model.SCVI, dict[str, float]]:
    """Train (or reload cached) batch-corrected scVI VAE and compute latent UMAP + Leiden."""
    reset_vram_stats()
    start_time = time.perf_counter()
    model_path = artifact_dir / "scvi_model"

    vae: scvi.model.SCVI | None = None
    if not retrain and (model_path / "model.pt").exists():
        try:
            logger.info("Found existing scVI checkpoint at %s; loading weights ...", model_path)
            vae = scvi.model.SCVI.load(str(model_path), adata=adata)
        except Exception as exc:
            logger.warning("Could not reload cached scVI model (%s); training from scratch.", exc)
            vae = None

    if vae is None:
        logger.info("Setting up scVI AnnData (layer='counts', batch_key='batch') ...")
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        vae = scvi.model.SCVI(
            adata,
            n_layers=2,
            n_hidden=128,
            n_latent=10,
            gene_likelihood="nb",
        )

        logger.info("Training scVI (max_epochs=%d, batch_size=%d) ...", max_epochs, batch_size)
        vae.train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping=True,
            check_val_every_n_epoch=5,
        )
        vae.save(str(model_path), overwrite=True)

    adata.obsm["X_scVI"] = vae.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata, random_state=42)
    adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()
    sc.tl.leiden(
        adata,
        key_added="leiden_scVI",
        resolution=leiden_resolution,
        flavor="igraph",
        n_iterations=2,
        random_state=42,
    )

    stats = {
        "runtime_sec": round(time.perf_counter() - start_time, 2),
        "peak_vram_mb": get_peak_vram_mb(),
    }
    logger.info("scVI finished in %.2fs | Peak VRAM: %.2f MB", stats["runtime_sec"], stats["peak_vram_mb"])
    return vae, stats


def _prepare_celltypist_adata(
    adata: sc.AnnData,
    data_dir: Path,
    model_name: str = "Immune_All_Low.pkl",
) -> sc.AnnData:
    """Build a log1p-normalized AnnData indexed by HGNC gene symbols verified against CellTypist."""
    raw_adata = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    ct_model = ct_models.Model.load(model_name)
    model_features = set(ct_model.features)

    # 1. Check if current var_names already overlap with CellTypist model features
    current_overlap = len(set(raw_adata.var_names.astype(str)) & model_features)
    if current_overlap >= 100:
        logger.info("CellTypist feature check: %d overlapping genes in var_names.", current_overlap)
        raw_adata.var_names_make_unique()
        return raw_adata

    # 2. Search every column in raw_adata.var for maximum overlap with CellTypist features
    best_symbols: np.ndarray | None = None
    best_overlap = 0
    best_source = ""

    for col in raw_adata.var.columns:
        candidates = raw_adata.var[col].astype(str).str.strip().to_numpy()
        overlap = len(set(candidates) & model_features)
        if overlap > best_overlap:
            best_overlap = overlap
            best_symbols = candidates
            best_source = f"adata.raw.var['{col}']"

    # 3. Search gene_info_pbmc.csv (matching by integer index or Ensembl ID)
    gene_info_path = data_dir / "gene_info_pbmc.csv"
    if best_overlap < 100 and gene_info_path.exists():
        gene_info = pd.read_csv(gene_info_path)
        for col in gene_info.columns:
            col_vals = gene_info[col].astype(str).str.strip()
            if len(set(col_vals) & model_features) < 100:
                continue

            # Case A: var_names are integer row indices ("0", "1", ...)
            if raw_adata.var_names.astype(str).str.isdigit().all():
                idx = raw_adata.var_names.astype(int).to_numpy()
                if idx.max() < len(gene_info):
                    candidates = col_vals.iloc[idx].to_numpy()
                    overlap = len(set(candidates) & model_features)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_symbols = candidates
                        best_source = f"{gene_info_path.name}['{col}'] (by index)"

            # Case B: Match via Ensembl ID column (e.g., ENSG...)
            for id_col in gene_info.columns:
                if id_col == col:
                    continue
                id_map = dict(zip(gene_info[id_col].astype(str).str.strip(), col_vals))
                # Check var_names or any var column for Ensembl ID matches
                series_to_check = [pd.Series(raw_adata.var_names.astype(str))] + [
                    raw_adata.var[vcol].astype(str) for vcol in raw_adata.var.columns
                ]
                for ser in series_to_check:
                    mapped = ser.map(id_map).fillna(ser).to_numpy()
                    overlap = len(set(mapped) & model_features)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_symbols = mapped
                        best_source = f"{gene_info_path.name}['{col}'] (via '{id_col}')"

    # 4. Fallback: read 10x genes.tsv / features.tsv(.gz) directly from extracted pbmc8k/pbmc4k
    if best_overlap < 100:
        for pattern in ("genes.tsv*", "features.tsv*"):
            for tsv_path in data_dir.rglob(pattern):
                tsv_df = pd.read_csv(tsv_path, sep="	", header=None)
                if tsv_df.shape[1] >= 2:
                    ensg_to_sym = dict(
                        zip(
                            tsv_df.iloc[:, 0].astype(str).str.strip(),
                            tsv_df.iloc[:, 1].astype(str).str.strip(),
                        )
                    )
                    series_to_check = [pd.Series(raw_adata.var_names.astype(str))] + [
                        raw_adata.var[vcol].astype(str) for vcol in raw_adata.var.columns
                    ]
                    for ser in series_to_check:
                        mapped = ser.map(ensg_to_sym).fillna(ser).to_numpy()
                        overlap = len(set(mapped) & model_features)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_symbols = mapped
                            best_source = f"{tsv_path.name} (Ensembl -> Symbol)"

    if best_symbols is None or best_overlap < 100:
        raise RuntimeError(
            f"Unable to map PBMC features to HGNC gene symbols for CellTypist. "
            f"var columns={list(raw_adata.var.columns)}, sample var_names={list(raw_adata.var_names[:5])}"
        )

    raw_adata.var_names = pd.Index(best_symbols)
    raw_adata.var_names_make_unique()
    logger.info(
        "Mapped CellTypist features via %s (%d overlapping reference genes).",
        best_source,
        best_overlap,
    )
    return raw_adata


def run_celltypist_reference(
    adata: sc.AnnData,
    data_dir: Path,
    model_name: str = "Immune_All_Low.pkl",
) -> dict[str, float]:
    """Run zero-shot CellTypist immune atlas annotation with majority voting."""
    start_time = time.perf_counter()
    logger.info("Downloading/loading CellTypist model: %s ...", model_name)
    ct_models.download_models(model=model_name)

    ct_adata = _prepare_celltypist_adata(adata, data_dir=data_dir, model_name=model_name)
    logger.info("Annotating cells with CellTypist (majority_voting=True) ...")
    preds = celltypist.annotate(
        ct_adata,
        model=model_name,
        majority_voting=True,
    )
    raw_labels = preds.predicted_labels["majority_voting"].astype(str).to_numpy()
    adata.obs["celltypist_raw"] = pd.Categorical(raw_labels)
    adata.obs["celltypist_pred"] = pd.Categorical(harmonize_celltypist_labels(raw_labels))

    stats = {
        "runtime_sec": round(time.perf_counter() - start_time, 2),
        "peak_vram_mb": 0.0,
    }
    logger.info("CellTypist finished in %.2fs (CPU)", stats["runtime_sec"])
    return stats


def create_semi_supervised_split(
    adata: sc.AnnData,
    mask_fraction: float = 0.5,
    seed: int = 42,
) -> None:
    """Create a shared 50/50 train/test split for scANVI and LatentMLP evaluation."""
    rng = np.random.default_rng(seed)
    true_labels = adata.obs["str_labels"].astype(str).to_numpy()
    is_test = rng.random(len(true_labels)) < mask_fraction

    masked_labels = true_labels.copy()
    masked_labels[is_test] = "Unknown"

    adata.obs["is_test_split"] = is_test
    adata.obs["semi_sup_labels"] = pd.Categorical(masked_labels)
    logger.info(
        "Created semi-supervised split: %d labeled train cells, %d held-out test cells.",
        int((~is_test).sum()),
        int(is_test.sum()),
    )


def train_scanvi_classifier(
    vae: scvi.model.SCVI,
    adata: sc.AnnData,
    artifact_dir: Path,
    max_epochs: int = 25,
    batch_size: int = 128,
) -> dict[str, float]:
    """Fine-tune semi-supervised scANVI classifier from pretrained scVI weights."""
    reset_vram_stats()
    start_time = time.perf_counter()

    logger.info("Initializing scANVI from pretrained scVI weights ...")
    lvae = scvi.model.SCANVI.from_scvi_model(
        vae,
        adata=adata,
        labels_key="semi_sup_labels",
        unlabeled_category="Unknown",
    )

    logger.info("Training scANVI (max_epochs=%d, batch_size=%d) ...", max_epochs, batch_size)
    lvae.train(max_epochs=max_epochs, batch_size=batch_size)
    adata.obs["scanvi_pred"] = pd.Categorical(np.asarray(lvae.predict(), dtype=str))

    model_path = artifact_dir / "scanvi_model"
    lvae.save(str(model_path), overwrite=True)

    stats = {
        "runtime_sec": round(time.perf_counter() - start_time, 2),
        "peak_vram_mb": get_peak_vram_mb(),
    }
    logger.info("scANVI finished in %.2fs | Peak VRAM: %.2f MB", stats["runtime_sec"], stats["peak_vram_mb"])
    return stats


def train_latent_mlp(
    adata: sc.AnnData,
    artifact_dir: Path,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict[str, float]:
    """Train supervised PyTorch MLP on X_scVI using the labeled 50% split."""
    reset_vram_stats()
    start_time = time.perf_counter()
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_latent = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    y_str = adata.obs["str_labels"].astype(str).to_numpy()
    is_test = adata.obs["is_test_split"].to_numpy(dtype=bool)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_str)

    X_train = torch.tensor(X_latent[~is_test], dtype=torch.float32)
    y_train = torch.tensor(y_all[~is_test], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    model = LatentMLP(
        in_dim=X_latent.shape[1],
        num_classes=len(encoder.classes_),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor(X_latent, dtype=torch.float32, device=device))
        pred_indices = logits.argmax(dim=1).cpu().numpy()

    adata.obs["mlp_pred"] = pd.Categorical(encoder.inverse_transform(pred_indices))

    ckpt_path = artifact_dir / "mlp_classifier.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": encoder.classes_.tolist(),
            "in_dim": int(X_latent.shape[1]),
        },
        ckpt_path,
    )

    stats = {
        "runtime_sec": round(time.perf_counter() - start_time, 2),
        "peak_vram_mb": get_peak_vram_mb(),
    }
    logger.info("LatentMLP finished in %.2fs | Peak VRAM: %.2f MB", stats["runtime_sec"], stats["peak_vram_mb"])
    return stats


def run_models_pipeline(
    input_path: str | Path = "data/pbmc_preprocessed.h5ad",
    output_path: str | Path = "data/pbmc_integrated.h5ad",
    artifact_dir: str | Path = "artifacts",
    scvi_epochs: int = 100,
    scanvi_epochs: int = 25,
    mlp_epochs: int = 50,
    batch_size: int = 128,
    retrain_scvi: bool = False,
) -> sc.AnnData:
    """Execute scVI integration and three-way classification benchmark."""
    scvi.settings.seed = 42
    in_file = Path(input_path)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    logger.info("Reading preprocessed AnnData from %s ...", in_file)
    adata = sc.read_h5ad(in_file)

    metrics: dict[str, dict[str, float]] = {}
    vae, metrics["scVI"] = train_scvi_integration(
        adata,
        artifact_dir=artifacts,
        max_epochs=scvi_epochs,
        batch_size=batch_size,
        retrain=retrain_scvi,
    )
    metrics["CellTypist"] = run_celltypist_reference(adata, data_dir=in_file.parent)

    create_semi_supervised_split(adata, mask_fraction=0.5, seed=42)
    metrics["scANVI"] = train_scanvi_classifier(
        vae,
        adata,
        artifact_dir=artifacts,
        max_epochs=scanvi_epochs,
        batch_size=batch_size,
    )
    metrics["LatentMLP"] = train_latent_mlp(
        adata,
        artifact_dir=artifacts,
        epochs=mlp_epochs,
        batch_size=batch_size,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_file)

    metrics_path = artifacts / "runtime_vram_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved integrated AnnData to %s and runtime stats to %s", out_file, metrics_path)
    return adata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train scVI, CellTypist, scANVI, and LatentMLP models."
    )
    parser.add_argument("--input", type=str, default="data/pbmc_preprocessed.h5ad")
    parser.add_argument("--output", type=str, default="data/pbmc_integrated.h5ad")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--scvi-epochs", type=int, default=100)
    parser.add_argument("--scanvi-epochs", type=int, default=25)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retrain-scvi", action="store_true", help="Force retraining scVI from scratch.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models_pipeline(
        input_path=args.input,
        output_path=args.output,
        artifact_dir=args.artifact_dir,
        scvi_epochs=args.scvi_epochs,
        scanvi_epochs=args.scanvi_epochs,
        mlp_epochs=args.mlp_epochs,
        batch_size=args.batch_size,
        retrain_scvi=args.retrain_scvi,
    )
