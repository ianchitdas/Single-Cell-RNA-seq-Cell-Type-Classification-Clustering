"""
preprocess.py
-------------
End-to-end single-cell RNA-seq preprocessing pipeline for multi-batch PBMC data.
Performs:
  1. Resilient dataset loading (`scvi.data.pbmc_dataset`) with automatic archive
     verification/extraction and corrupt-cache recovery on Windows.
  2. Raw count preservation in `adata.layers["counts"]` for negative-binomial modeling.
  3. Quality control filtering (gene count & mitochondrial fraction thresholds).
  4. Per-batch doublet detection and removal via Scrublet.
  5. Library-size normalization (1e4) + log1p transform stored in `adata.raw`.
  6. Batch-aware Highly Variable Gene (HVG) selection (top 2,000 genes).
  7. Unsupervised baseline dimensionality reduction and clustering (PCA -> KNN -> UMAP -> Leiden).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import tarfile

import scanpy as sc
import scvi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _repair_or_extract_10x_archives(data_dir: Path) -> None:
    """Ensure cached 10x tar.gz archives are valid and extracted, or purge broken ones."""
    for sub in ("pbmc8k", "pbmc4k"):
        sub_dir = data_dir / sub
        if not sub_dir.exists():
            continue

        has_mtx = any(sub_dir.rglob("matrix.mtx*"))
        if has_mtx:
            continue

        tar_files = list(sub_dir.glob("*.tar.gz"))
        for tar_path in tar_files:
            try:
                logger.info("Extracting cached archive %s ...", tar_path)
                with tarfile.open(tar_path, "r:gz") as tf:
                    tf.extractall(path=sub_dir, filter="data")
            except (tarfile.TarError, EOFError, OSError) as exc:
                logger.warning(
                    "Corrupted or incomplete archive %s (%s). Removing so scvi can re-download.",
                    tar_path,
                    exc,
                )
                shutil.rmtree(sub_dir, ignore_errors=True)
                break


def load_pbmc_data(data_dir: Path) -> sc.AnnData:
    """Load the two-batch 10x PBMC dataset from scvi-tools and preserve raw counts."""
    data_dir.mkdir(parents=True, exist_ok=True)
    _repair_or_extract_10x_archives(data_dir)

    logger.info("Loading scvi.data.pbmc_dataset into %s ...", data_dir)
    try:
        adata = scvi.data.pbmc_dataset(save_path=str(data_dir))
    except FileNotFoundError:
        logger.warning(
            "Missing matrix.mtx after initial load; clearing cached pbmc8k/pbmc4k folders and retrying ..."
        )
        for sub in ("pbmc8k", "pbmc4k"):
            shutil.rmtree(data_dir / sub, ignore_errors=True)
        adata = scvi.data.pbmc_dataset(save_path=str(data_dir))

    # Ensure categorical string labels for downstream consistency
    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    adata.obs["str_labels"] = adata.obs["str_labels"].astype(str).astype("category")

    # Preserve raw integer UMI counts prior to normalization
    adata.layers["counts"] = adata.X.copy()
    logger.info("Loaded AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def run_qc_and_doublet_removal(
    adata: sc.AnnData,
    min_genes: int = 200,
    max_mt_pct: float = 15.0,
) -> sc.AnnData:
    """Filter low-quality cells and remove predicted doublets per batch using Scrublet."""
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    n_before = adata.n_obs
    qc_mask = (adata.obs["n_genes_by_counts"] > min_genes) & (
        adata.obs["pct_counts_mt"] < max_mt_pct
    )
    adata = adata[qc_mask].copy()
    logger.info(
        "QC filter (genes > %d, MT < %.1f%%): retained %d / %d cells",
        min_genes,
        max_mt_pct,
        adata.n_obs,
        n_before,
    )

    # Batch-aware Scrublet doublet detection
    logger.info("Running batch-aware Scrublet doublet detection ...")
    sc.pp.scrublet(adata, batch_key="batch")
    doublet_mask = adata.obs["predicted_doublet"].fillna(False).astype(bool)
    n_doublets = int(doublet_mask.sum())
    adata = adata[~doublet_mask].copy()
    logger.info(
        "Removed %d predicted doublets; %d singlets remaining.",
        n_doublets,
        adata.n_obs,
    )
    return adata


def normalize_and_select_hvgs(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
) -> sc.AnnData:
    """Normalize counts for Scanpy/CellTypist and select batch-aware HVGs."""
    logger.info("Normalizing total counts (target_sum=1e4) and applying log1p ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Freeze full log-normalized transcriptome in .raw for CellTypist & marker genes
    adata.raw = adata

    logger.info("Selecting top %d batch-aware Highly Variable Genes ...", n_top_genes)
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            layer="counts",
            batch_key="batch",
            subset=True,
        )
    except ImportError:
        logger.warning(
            "scikit-misc not found; falling back to flavor='seurat' on log1p counts."
        )
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat",
            batch_key="batch",
            subset=True,
        )

    logger.info("Subsetted AnnData shape: %d cells x %d HVGs", adata.n_obs, adata.n_vars)
    return adata


def run_pca_baseline(
    adata: sc.AnnData,
    n_pcs: int = 30,
    leiden_resolution: float = 0.5,
) -> sc.AnnData:
    """Compute unsupervised PCA, neighborhood graph, UMAP, and Leiden clusters."""
    logger.info("Running PCA baseline (n_comps=%d) and Leiden clustering ...", n_pcs)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep="X_pca")
    sc.tl.umap(adata, random_state=42)
    adata.obsm["X_umap_pca"] = adata.obsm["X_umap"].copy()

    sc.tl.leiden(
        adata,
        key_added="leiden_pca",
        resolution=leiden_resolution,
        flavor="igraph",
        n_iterations=2,
        random_state=42,
    )
    logger.info(
        "PCA Leiden clustering complete: %d clusters identified.",
        adata.obs["leiden_pca"].nunique(),
    )
    return adata


def run_preprocessing_pipeline(
    data_dir: str | Path = "data",
    output_path: str | Path = "data/pbmc_preprocessed.h5ad",
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    leiden_resolution: float = 0.5,
) -> sc.AnnData:
    """Execute full preprocessing pipeline and save AnnData to disk."""
    sc.settings.verbosity = 1
    if hasattr(sc, "set_figure_params"):
        sc.set_figure_params(dpi=120, frameon=False)
    else:
        sc.settings.set_figure_params(dpi=120, frameon=False)

    adata = load_pbmc_data(Path(data_dir))
    adata = run_qc_and_doublet_removal(adata)
    adata = normalize_and_select_hvgs(adata, n_top_genes=n_top_genes)
    adata = run_pca_baseline(
        adata,
        n_pcs=n_pcs,
        leiden_resolution=leiden_resolution,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_file)
    logger.info("Saved preprocessed AnnData to %s", out_file)
    return adata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess PBMC scRNA-seq data and run PCA + Leiden baseline."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="data/pbmc_preprocessed.h5ad")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        leiden_resolution=args.resolution,
    )
