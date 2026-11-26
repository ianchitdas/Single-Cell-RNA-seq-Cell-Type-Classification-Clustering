"""
evaluate.py
-----------
Quantitative benchmarking and publication-ready figure generation for single-cell
clustering, batch integration, marker validation, and cell-type classification.

Generates:
  1. Benchmark CSVs & Markdown Tables (`results/`):
     - `clustering_benchmark.csv`: ARI, NMI, Cell-Type ASW, and Batch Mixing (PCA vs. scVI).
     - `classification_benchmark.csv`: Accuracy, Macro-F1, Weighted-F1, Runtime, and Peak VRAM
       comparing CellTypist, scANVI, and LatentMLP on held-out test cells.
  2. Visualizations (`figures/`):
     - `umap_batch_integration.png`: 2x3 UMAP grid comparing PCA vs. scVI (Batch, Labels, Leiden).
     - `umap_classifier_predictions.png`: 2x2 UMAP grid comparing Ground Truth vs. all 3 classifiers.
     - `marker_genes_dotplot.png`: Canonical PBMC lineage marker expression across cell types.
     - `confusion_matrices.png`: Row-normalized confusion matrices for all 3 classifiers.
     - `benchmark_summary.png`: Grouped bar charts of clustering & classification metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CANONICAL_PBMC_MARKERS: dict[str, list[str]] = {
    "CD4/CD8 T": ["CD3D", "CD3E", "IL7R", "CD8A"],
    "B Cells": ["MS4A1", "CD79A"],
    "NK Cells": ["NKG7", "GNLY"],
    "Monocytes": ["CD14", "LYZ", "FCGR3A"],
    "DC / Megakaryo": ["CST3", "FCER1A", "PPBP"],
}


def _resolve_existing_path(primary: str | Path) -> Path:
    """Resolve path relative to CWD or parent directory (handles running from `source/`)."""
    p = Path(primary)
    if p.exists():
        return p
    alt = Path("..") / p
    if alt.exists():
        return alt
    return p


def _format_markdown_table(df: pd.DataFrame) -> str:
    """Render DataFrame as a Markdown table without requiring optional `tabulate`."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(c) for c in df.columns]
        sep = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in df.itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)


def _get_symbol_indexed_raw(adata: sc.AnnData, data_dir: Path) -> sc.AnnData:
    """Return `adata.raw` with HGNC gene symbols as `var_names` for marker plotting."""
    raw_adata = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    target_genes = {g for genes in CANONICAL_PBMC_MARKERS.values() for g in genes}

    # 1. Direct match on var_names
    best_symbols: np.ndarray | None = None
    best_overlap = len(set(raw_adata.var_names.astype(str)) & target_genes)
    best_source = "adata.raw.var_names"
    if best_overlap >= 5:
        raw_adata.var_names_make_unique()
        return raw_adata

    # 2. Direct match on any column in raw_adata.var
    for col in raw_adata.var.columns:
        candidates = raw_adata.var[col].astype(str).str.strip().to_numpy()
        overlap = len(set(candidates) & target_genes)
        if overlap > best_overlap:
            best_overlap = overlap
            best_symbols = candidates
            best_source = f"adata.raw.var['{col}']"

    # 3. Map via gene_info_pbmc.csv (integer index OR Ensembl ID -> symbol)
    gene_info_path = data_dir / "gene_info_pbmc.csv"
    if best_overlap < 5 and gene_info_path.exists():
        gene_info = pd.read_csv(gene_info_path)
        for col in gene_info.columns:
            col_vals = gene_info[col].astype(str).str.strip()
            if len(set(col_vals) & target_genes) < 5:
                continue

            if raw_adata.var_names.astype(str).str.isdigit().all():
                idx = raw_adata.var_names.astype(int).to_numpy()
                if idx.max() < len(gene_info):
                    candidates = col_vals.iloc[idx].to_numpy()
                    overlap = len(set(candidates) & target_genes)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_symbols = candidates
                        best_source = f"{gene_info_path.name}['{col}'] (by index)"

            for id_col in gene_info.columns:
                if id_col == col:
                    continue
                id_map = dict(zip(gene_info[id_col].astype(str).str.strip(), col_vals))
                series_to_check = [pd.Series(raw_adata.var_names.astype(str))] + [
                    raw_adata.var[vcol].astype(str) for vcol in raw_adata.var.columns
                ]
                for ser in series_to_check:
                    mapped = ser.map(id_map).fillna(ser).to_numpy()
                    overlap = len(set(mapped) & target_genes)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_symbols = mapped
                        best_source = f"{gene_info_path.name}['{col}'] (via '{id_col}')"

    # 4. Fallback: read 10x genes.tsv / features.tsv(.gz) from extracted pbmc8k/pbmc4k
    if best_overlap < 5:
        for pattern in ("genes.tsv*", "features.tsv*"):
            for tsv_path in data_dir.rglob(pattern):
                tsv_df = pd.read_csv(tsv_path, sep="\t", header=None)
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
                        overlap = len(set(mapped) & target_genes)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_symbols = mapped
                            best_source = f"{tsv_path.name} (Ensembl -> Symbol)"

    if best_symbols is not None and best_overlap >= 5:
        raw_adata.var_names = pd.Index(best_symbols)
        logger.info(
            "Mapped %d canonical PBMC markers via %s.",
            best_overlap,
            best_source,
        )

    raw_adata.var_names_make_unique()
    return raw_adata


def compute_clustering_benchmarks(adata: sc.AnnData) -> pd.DataFrame:
    """Evaluate clustering fidelity and batch integration on PCA vs. scVI embeddings."""
    true_labels = adata.obs["str_labels"].astype(str).to_numpy()
    batch_labels = adata.obs["batch"].astype(str).to_numpy()

    configs = [
        ("PCA + Leiden", adata.obsm["X_pca"][:, :30], adata.obs["leiden_pca"].astype(str).to_numpy()),
        ("scVI + Leiden", adata.obsm["X_scVI"], adata.obs["leiden_scVI"].astype(str).to_numpy()),
    ]

    rows = []
    for name, embedding, clusters in configs:
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        celltype_asw = silhouette_score(
            embedding,
            true_labels,
            sample_size=min(10000, len(true_labels)),
            random_state=42,
        )
        batch_asw = silhouette_score(
            embedding,
            batch_labels,
            sample_size=min(10000, len(batch_labels)),
            random_state=42,
        )
        batch_mixing = 1.0 - abs(float(batch_asw))

        rows.append(
            {
                "Representation": name,
                "Num_Clusters": int(pd.Series(clusters).nunique()),
                "ARI": round(float(ari), 4),
                "NMI": round(float(nmi), 4),
                "CellType_ASW": round(float(celltype_asw), 4),
                "Batch_Mixing (1-|ASW|)": round(batch_mixing, 4),
            }
        )

    return pd.DataFrame(rows)


def compute_classification_benchmarks(
    adata: sc.AnnData,
    runtime_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate CellTypist, scANVI, and LatentMLP on the held-out 50% test split."""
    is_test = adata.obs["is_test_split"].to_numpy(dtype=bool)
    y_true = adata.obs.loc[is_test, "str_labels"].astype(str).to_numpy()

    models = [
        ("CellTypist (Zero-Shot)", "celltypist_pred", "CellTypist"),
        ("scANVI (Semi-Supervised)", "scanvi_pred", "scANVI"),
        ("PyTorch MLP on X_scVI", "mlp_pred", "LatentMLP"),
    ]

    rows = []
    for display_name, obs_col, stats_key in models:
        y_pred = adata.obs.loc[is_test, obs_col].astype(str).to_numpy()
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        stat_entry = (runtime_stats or {}).get(stats_key, {})
        rows.append(
            {
                "Classifier": display_name,
                "Test_Accuracy": round(float(acc), 4),
                "Macro_F1": round(float(macro_f1), 4),
                "Weighted_F1": round(float(weighted_f1), 4),
                "Runtime_Sec": stat_entry.get("runtime_sec", np.nan),
                "Peak_VRAM_MB": stat_entry.get("peak_vram_mb", np.nan),
            }
        )

    return pd.DataFrame(rows)


def plot_umap_comparison(adata: sc.AnnData, output_file: Path) -> None:
    """Plot 2x3 UMAP grid comparing PCA baseline vs. scVI latent space."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 10.5), dpi=150)

    specs = [
        (0, 0, "X_umap_pca", "batch", "PCA UMAP — Batch"),
        (0, 1, "X_umap_pca", "str_labels", "PCA UMAP — Ground Truth Cell Types"),
        (0, 2, "X_umap_pca", "leiden_pca", "PCA UMAP — Leiden Clusters"),
        (1, 0, "X_umap_scvi", "batch", "scVI UMAP — Batch (Integrated)"),
        (1, 1, "X_umap_scvi", "str_labels", "scVI UMAP — Ground Truth Cell Types"),
        (1, 2, "X_umap_scvi", "leiden_scVI", "scVI UMAP — Leiden Clusters"),
    ]

    for row, col, obsm_key, obs_key, title in specs:
        ax = axes[row, col]
        adata.obsm["X_umap"] = adata.obsm[obsm_key]
        sc.pl.umap(
            adata,
            color=obs_key,
            ax=ax,
            show=False,
            title=title,
            frameon=False,
            legend_loc="on data" if "leiden" in obs_key else "right margin",
            legend_fontsize=8,
        )

    plt.subplots_adjust(wspace=0.42, hspace=0.25)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved UMAP batch integration figure to %s", output_file)


def plot_classifier_umaps(adata: sc.AnnData, output_file: Path) -> None:
    """Plot 2x2 UMAP grid comparing Ground Truth labels vs. all 3 classifiers on scVI space."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    adata.obsm["X_umap"] = adata.obsm["X_umap_scvi"]

    panels = [
        (0, 0, "str_labels", "Ground Truth Cell Types"),
        (0, 1, "celltypist_pred", "CellTypist (Zero-Shot Reference)"),
        (1, 0, "scanvi_pred", "scANVI (Semi-Supervised VAE)"),
        (1, 1, "mlp_pred", "PyTorch LatentMLP (Supervised)"),
    ]

    for row, col, obs_key, title in panels:
        sc.pl.umap(
            adata,
            color=obs_key,
            ax=axes[row, col],
            show=False,
            title=title,
            frameon=False,
            legend_loc="right margin",
            legend_fontsize=8,
        )

    plt.subplots_adjust(wspace=0.45, hspace=0.25)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved classifier UMAP comparison to %s", output_file)


def plot_marker_genes_dotplot(adata: sc.AnnData, data_dir: Path, output_file: Path) -> None:
    """Plot canonical PBMC lineage marker expression across ground-truth cell types."""
    sym_adata = _get_symbol_indexed_raw(adata, data_dir=data_dir)
    available_vars = set(sym_adata.var_names.astype(str))

    filtered_markers = {
        group: [g for g in genes if g in available_vars]
        for group, genes in CANONICAL_PBMC_MARKERS.items()
    }
    filtered_markers = {k: v for k, v in filtered_markers.items() if v}

    if not filtered_markers:
        logger.warning("No canonical PBMC markers found in var_names; skipping marker dotplot.")
        return

    dp = sc.pl.dotplot(
        sym_adata,
        var_names=filtered_markers,
        groupby="str_labels",
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    dp.make_figure()
    dp.fig.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved canonical PBMC marker dotplot to %s", output_file)


def plot_confusion_matrices(adata: sc.AnnData, output_file: Path) -> None:
    """Plot side-by-side row-normalized confusion matrices on held-out test cells."""
    is_test = adata.obs["is_test_split"].to_numpy(dtype=bool)
    y_true = adata.obs.loc[is_test, "str_labels"].astype(str).to_numpy()
    labels = sorted(pd.Series(y_true).unique())

    classifiers = [
        ("CellTypist (Zero-Shot)", "celltypist_pred"),
        ("scANVI (Semi-Supervised)", "scanvi_pred"),
        ("PyTorch MLP on X_scVI", "mlp_pred"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=150)
    for ax, (title, col_name) in zip(axes, classifiers):
        y_pred = adata.obs.loc[is_test, col_name].astype(str).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels if ax is axes[0] else False,
            cbar=ax is axes[-1],
            ax=ax,
            annot_kws={"size": 7.5},
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("True Cell Type", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrices to %s", output_file)


def plot_benchmark_bars(
    cluster_df: pd.DataFrame,
    class_df: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot summary bar charts for clustering and classification metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150)

    # Panel A: Clustering & Batch Integration metrics
    cluster_melted = cluster_df.melt(
        id_vars="Representation",
        value_vars=["ARI", "NMI", "CellType_ASW", "Batch_Mixing (1-|ASW|)"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(
        data=cluster_melted,
        x="Metric",
        y="Score",
        hue="Representation",
        palette=["#64748b", "#0f766e"],
        ax=axes[0],
    )
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("Unsupervised Clustering & Batch Integration", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Classification metrics
    class_melted = class_df.melt(
        id_vars="Classifier",
        value_vars=["Test_Accuracy", "Macro_F1", "Weighted_F1"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(
        data=class_melted,
        x="Metric",
        y="Score",
        hue="Classifier",
        palette=["#94a3b8", "#2563eb", "#0d9488"],
        ax=axes[1],
    )
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("Held-Out Cell Type Classification (50% Test Split)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved benchmark bar chart to %s", output_file)


def run_evaluation_pipeline(
    input_path: str | Path = "data/pbmc_integrated.h5ad",
    metrics_json_path: str | Path = "artifacts/runtime_vram_metrics.json",
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all benchmark tables, save CSVs, and render all 5 figures."""
    resolved_input = _resolve_existing_path(input_path)
    resolved_metrics = _resolve_existing_path(metrics_json_path)

    results_out = Path(results_dir)
    figures_out = Path(figures_dir)
    results_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading integrated AnnData from %s ...", resolved_input)
    adata = sc.read_h5ad(resolved_input)

    runtime_stats = None
    if resolved_metrics.exists():
        runtime_stats = json.loads(resolved_metrics.read_text())

    cluster_df = compute_clustering_benchmarks(adata)
    class_df = compute_classification_benchmarks(adata, runtime_stats=runtime_stats)

    cluster_df.to_csv(results_out / "clustering_benchmark.csv", index=False)
    class_df.to_csv(results_out / "classification_benchmark.csv", index=False)

    plot_umap_comparison(adata, figures_out / "umap_batch_integration.png")
    plot_classifier_umaps(adata, figures_out / "umap_classifier_predictions.png")
    plot_marker_genes_dotplot(
        adata,
        data_dir=resolved_input.parent,
        output_file=figures_out / "marker_genes_dotplot.png",
    )
    plot_confusion_matrices(adata, figures_out / "confusion_matrices.png")
    plot_benchmark_bars(cluster_df, class_df, figures_out / "benchmark_summary.png")

    print("\n=== CLUSTERING & BATCH INTEGRATION BENCHMARK ===")
    print(_format_markdown_table(cluster_df))
    print("\n=== THREE-WAY CLASSIFICATION BENCHMARK (HELD-OUT TEST SPLIT) ===")
    print(_format_markdown_table(class_df))

    return cluster_df, class_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-cell clustering and classification benchmarks."
    )
    parser.add_argument("--input", type=str, default="data/pbmc_integrated.h5ad")
    parser.add_argument("--metrics-json", type=str, default="artifacts/runtime_vram_metrics.json")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation_pipeline(
        input_path=args.input,
        metrics_json_path=args.metrics_json,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
