"""Differential transcript usage analysis for single-cell long-read data.

Detects genes where different cell clusters preferentially use different
transcript isoforms (isoform switching). Implements chi-squared and
Dirichlet-multinomial tests for differential transcript usage (DTU),
with Benjamini-Hochberg multiple testing correction. Designed for
transcript-level count matrices from long-read single-cell platforms.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import scipy.stats

from ._logging import get_named_logger, wf_parser


def argparser():
    """Create argument parser."""
    parser = wf_parser("differential_transcript_usage")

    parser.add_argument(
        "transcript_matrix_dir", type=Path,
        help="Path to processed transcript-level MEX matrix directory.")
    parser.add_argument(
        "--clusters", type=Path, required=True,
        help="TSV with columns barcode, cluster.")
    parser.add_argument(
        "--gene_transcript_map", type=Path, required=True,
        help="TSV mapping transcript_id to gene_id.")
    parser.add_argument(
        "--output_dtu", type=Path, default="dtu_results.tsv",
        help="Output TSV with DTU test results per gene per comparison.")
    parser.add_argument(
        "--output_switching", type=Path, default="isoform_switching.tsv",
        help="Output TSV with detected isoform switching events.")
    parser.add_argument(
        "--output_summary", type=Path, default="dtu_summary.json",
        help="Output JSON with DTU analysis summary statistics.")
    parser.add_argument(
        "--test_method", default="chi_squared",
        choices=["chi_squared", "dirichlet_multinomial"],
        help="Statistical test for DTU.")
    parser.add_argument(
        "--min_cells_per_cluster", type=int, default=10,
        help="Minimum cells in a cluster to include in testing.")
    parser.add_argument(
        "--min_gene_counts", type=int, default=20,
        help="Minimum total transcript counts per gene to test.")
    parser.add_argument(
        "--min_isoforms", type=int, default=2,
        help="Minimum number of expressed isoforms per gene.")
    parser.add_argument(
        "--fdr_threshold", type=float, default=0.05,
        help="FDR threshold for significance.")
    parser.add_argument(
        "--cluster_column", type=str, default="cluster",
        help="Column name for cluster labels in the clusters TSV.")

    return parser


def load_gene_transcript_map(map_file):
    """Read a two-column TSV mapping transcript_id to gene_id.

    :param map_file: Path to TSV with columns transcript_id, gene_id.
    :returns: tuple of (transcript_to_gene dict, gene_to_transcripts dict).
    """
    logger = get_named_logger("DTU_Map")
    df = pd.read_csv(map_file, sep='\t', header=0)
    cols = df.columns.tolist()
    if len(cols) < 2:
        raise ValueError(
            f"Gene-transcript map must have at least 2 columns, got {cols}")
    tx_col, gene_col = cols[0], cols[1]

    transcript_to_gene = dict(zip(df[tx_col], df[gene_col]))
    gene_to_transcripts = {}
    for tx, gene in transcript_to_gene.items():
        gene_to_transcripts.setdefault(gene, []).append(tx)

    logger.info(
        f"Loaded map: {len(transcript_to_gene)} transcripts, "
        f"{len(gene_to_transcripts)} genes.")
    return transcript_to_gene, gene_to_transcripts


def build_gene_groups(transcript_names, gene_map, min_isoforms=2):
    """Group transcript indices by gene and filter by isoform count.

    :param transcript_names: array-like of transcript names from the matrix.
    :param gene_map: dict mapping transcript_id to gene_id.
    :param min_isoforms: minimum number of transcripts per gene.
    :returns: dict of gene_id -> list of transcript indices.
    """
    gene_groups = {}
    for idx, tx_name in enumerate(transcript_names):
        gene_id = gene_map.get(tx_name)
        if gene_id is not None:
            gene_groups.setdefault(gene_id, []).append(idx)

    # Filter to genes with enough isoforms
    filtered = {
        gene: indices for gene, indices in gene_groups.items()
        if len(indices) >= min_isoforms
    }
    return filtered


def chi_squared_dtu_test(counts_a, counts_b):
    """Chi-squared test for differential transcript usage.

    Builds a 2 x n_transcripts contingency table and runs chi2_contingency.

    :param counts_a: 1D array of transcript counts for cluster A.
    :param counts_b: 1D array of transcript counts for cluster B.
    :returns: tuple of (chi2_statistic, pvalue, effect_size).
    """
    counts_a = np.asarray(counts_a, dtype=float)
    counts_b = np.asarray(counts_b, dtype=float)

    # Remove transcripts with zero counts in both groups
    nonzero = (counts_a + counts_b) > 0
    if nonzero.sum() < 2:
        return np.nan, np.nan, 0.0

    counts_a = counts_a[nonzero]
    counts_b = counts_b[nonzero]

    contingency = np.array([counts_a, counts_b])
    n_total = contingency.sum()
    if n_total == 0:
        return np.nan, np.nan, 0.0

    try:
        chi2, pvalue, dof, _ = scipy.stats.chi2_contingency(contingency)
    except ValueError:
        return np.nan, np.nan, 0.0

    # Cramér's V as effect size
    n_rows, n_cols = contingency.shape
    min_dim = min(n_rows - 1, n_cols - 1)
    if min_dim == 0 or n_total == 0:
        effect_size = 0.0
    else:
        effect_size = np.sqrt(chi2 / (n_total * min_dim))

    return chi2, pvalue, effect_size


def _dm_loglikelihood(alpha, counts_matrix):
    """Compute Dirichlet-multinomial log-likelihood.

    :param alpha: 1D array of Dirichlet parameters.
    :param counts_matrix: 2D array (n_cells x n_transcripts).
    :returns: float log-likelihood.
    """
    from scipy.special import gammaln

    alpha = np.maximum(alpha, 1e-10)
    alpha_sum = alpha.sum()
    n_cells, n_tx = counts_matrix.shape
    ll = 0.0
    for i in range(n_cells):
        x = counts_matrix[i]
        n_i = x.sum()
        ll += gammaln(alpha_sum) - gammaln(n_i + alpha_sum)
        for j in range(n_tx):
            ll += gammaln(x[j] + alpha[j]) - gammaln(alpha[j])
    return ll


def _dm_neg_loglikelihood(log_alpha, counts_matrix):
    """Negative log-likelihood for optimization (works in log-space)."""
    alpha = np.exp(log_alpha)
    return -_dm_loglikelihood(alpha, counts_matrix)


def dirichlet_multinomial_test(counts_a, counts_b):
    """Dirichlet-multinomial likelihood ratio test for DTU.

    Fits DM models to each group separately and to the combined data,
    then computes a likelihood ratio statistic.

    :param counts_a: 2D array (n_cells_a x n_transcripts) per-cell counts.
    :param counts_b: 2D array (n_cells_b x n_transcripts) per-cell counts.
    :returns: tuple of (lr_statistic, pvalue, effect_size).
    """
    from scipy.optimize import minimize

    counts_a = np.asarray(counts_a, dtype=float)
    counts_b = np.asarray(counts_b, dtype=float)

    if counts_a.ndim == 1:
        counts_a = counts_a.reshape(1, -1)
    if counts_b.ndim == 1:
        counts_b = counts_b.reshape(1, -1)

    n_tx = counts_a.shape[1]
    if n_tx < 2:
        return np.nan, np.nan, 0.0

    # Remove all-zero transcripts across both groups
    nonzero = (counts_a.sum(axis=0) + counts_b.sum(axis=0)) > 0
    if nonzero.sum() < 2:
        return np.nan, np.nan, 0.0

    counts_a = counts_a[:, nonzero]
    counts_b = counts_b[:, nonzero]
    n_tx = counts_a.shape[1]
    combined = np.vstack([counts_a, counts_b])

    try:
        # Initial alpha from method of moments
        init_alpha = np.ones(n_tx)

        # Fit shared model (H0)
        res_shared = minimize(
            _dm_neg_loglikelihood, np.log(init_alpha), args=(combined,),
            method='L-BFGS-B', options={'maxiter': 200})
        ll_shared = -res_shared.fun

        # Fit separate models (H1)
        res_a = minimize(
            _dm_neg_loglikelihood, np.log(init_alpha), args=(counts_a,),
            method='L-BFGS-B', options={'maxiter': 200})
        ll_a = -res_a.fun

        res_b = minimize(
            _dm_neg_loglikelihood, np.log(init_alpha), args=(counts_b,),
            method='L-BFGS-B', options={'maxiter': 200})
        ll_b = -res_b.fun

        # Likelihood ratio statistic
        lr_stat = 2 * ((ll_a + ll_b) - ll_shared)
        lr_stat = max(lr_stat, 0.0)

        # Chi-squared approximation
        df = n_tx - 1
        pvalue = scipy.stats.chi2.sf(lr_stat, df)

        # Effect size: normalized LR
        n_total = combined.sum()
        effect_size = np.sqrt(lr_stat / n_total) if n_total > 0 else 0.0

        return lr_stat, pvalue, effect_size

    except Exception:
        # Fall back to chi-squared test on aggregated counts
        agg_a = counts_a.sum(axis=0)
        agg_b = counts_b.sum(axis=0)
        return chi_squared_dtu_test(agg_a, agg_b)


def correct_pvalues(pvalues, method='fdr_bh'):
    """Benjamini-Hochberg FDR correction.

    Handles NaN p-values by skipping them and preserving their position.

    :param pvalues: array-like of p-values (may contain NaN).
    :param method: correction method (only 'fdr_bh' implemented).
    :returns: array of adjusted p-values.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)

    # Identify valid (non-NaN) p-values
    valid_mask = ~np.isnan(pvalues)
    if valid_mask.sum() == 0:
        return adjusted

    valid_pvals = pvalues[valid_mask]
    n = len(valid_pvals)

    # Sort p-values
    sort_idx = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sort_idx]

    # BH adjustment: p_adj[i] = p[i] * n / rank[i]
    ranks = np.arange(1, n + 1)
    adjusted_sorted = sorted_pvals * n / ranks

    # Enforce monotonicity (from largest to smallest)
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Cap at 1.0
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Unsort
    unsort_idx = np.argsort(sort_idx)
    valid_adjusted = adjusted_sorted[unsort_idx]
    adjusted[valid_mask] = valid_adjusted

    return adjusted


def detect_isoform_switching(counts_a, counts_b, transcript_names):
    """Detect isoform switching between two groups.

    Compares transcript proportions and identifies cases where the
    dominant isoform differs between clusters.

    :param counts_a: 1D array of aggregated transcript counts, cluster A.
    :param counts_b: 1D array of aggregated transcript counts, cluster B.
    :param transcript_names: list of transcript names for this gene.
    :returns: dict with switching details, or None if no switching detected.
    """
    counts_a = np.asarray(counts_a, dtype=float)
    counts_b = np.asarray(counts_b, dtype=float)

    total_a = counts_a.sum()
    total_b = counts_b.sum()

    if total_a == 0 or total_b == 0:
        return None

    prop_a = counts_a / total_a
    prop_b = counts_b / total_b

    dominant_idx_a = np.argmax(prop_a)
    dominant_idx_b = np.argmax(prop_b)

    if dominant_idx_a == dominant_idx_b:
        return None

    # Switching score: change in proportion of cluster A's dominant isoform
    switching_score = abs(
        prop_a[dominant_idx_a] - prop_b[dominant_idx_a])

    return {
        'dominant_transcript_a': transcript_names[dominant_idx_a],
        'proportion_a': float(prop_a[dominant_idx_a]),
        'dominant_transcript_b': transcript_names[dominant_idx_b],
        'proportion_b': float(prop_b[dominant_idx_b]),
        'switching_score': float(switching_score),
    }


def main(args):
    """Run differential transcript usage analysis."""
    import scanpy as sc

    logger = get_named_logger("DTU")
    logger.info("Starting differential transcript usage analysis.")

    # Load transcript-level matrix
    logger.info(f"Loading transcript matrix from {args.transcript_matrix_dir}.")
    adata = sc.read_10x_mtx(
        str(args.transcript_matrix_dir), var_names='gene_symbols')
    adata.var_names_make_unique()
    logger.info(
        f"Loaded matrix: {adata.shape[0]} cells x "
        f"{adata.shape[1]} transcripts.")

    # Load cluster assignments
    clusters_df = pd.read_csv(args.clusters, sep='\t')
    cluster_col = args.cluster_column
    if cluster_col not in clusters_df.columns:
        raise ValueError(
            f"Cluster column '{cluster_col}' not found in {args.clusters}. "
            f"Available columns: {clusters_df.columns.tolist()}")

    # Map barcodes to clusters
    barcode_col = clusters_df.columns[0]
    barcode_to_cluster = dict(
        zip(clusters_df[barcode_col], clusters_df[cluster_col]))

    # Subset to cells present in both matrix and cluster file
    common_barcodes = [
        bc for bc in adata.obs_names if bc in barcode_to_cluster]
    if len(common_barcodes) == 0:
        raise ValueError("No overlapping barcodes between matrix and clusters.")

    adata = adata[common_barcodes].copy()
    adata.obs['cluster'] = [
        str(barcode_to_cluster[bc]) for bc in adata.obs_names]
    logger.info(
        f"Matched {adata.shape[0]} cells with cluster assignments.")

    # Filter clusters by minimum cell count
    cluster_counts = pd.Series(adata.obs['cluster']).value_counts()
    valid_clusters = cluster_counts[
        cluster_counts >= args.min_cells_per_cluster].index.tolist()
    if len(valid_clusters) < 2:
        logger.warning(
            f"Only {len(valid_clusters)} clusters have >= "
            f"{args.min_cells_per_cluster} cells. "
            "Cannot perform DTU analysis.")
        # Write empty outputs
        pd.DataFrame(columns=[
            'gene', 'cluster_a', 'cluster_b', 'test_statistic',
            'pvalue', 'pvalue_adj', 'n_transcripts', 'effect_size'
        ]).to_csv(args.output_dtu, sep='\t', index=False)
        pd.DataFrame(columns=[
            'gene', 'cluster_a', 'cluster_b', 'dominant_transcript_a',
            'proportion_a', 'dominant_transcript_b', 'proportion_b',
            'switching_score'
        ]).to_csv(args.output_switching, sep='\t', index=False)
        with open(args.output_summary, 'w') as fh:
            json.dump({
                'n_genes_tested': 0, 'n_significant': 0,
                'n_switching_events': 0, 'top_dtu_genes': []
            }, fh, indent=2)
        return

    adata = adata[adata.obs['cluster'].isin(valid_clusters)].copy()
    logger.info(
        f"Testing {len(valid_clusters)} clusters with >= "
        f"{args.min_cells_per_cluster} cells each.")

    # Load gene-transcript map
    transcript_to_gene, _ = load_gene_transcript_map(args.gene_transcript_map)

    # Build gene groups from transcripts in the matrix
    transcript_names = np.array(adata.var_names)
    gene_groups = build_gene_groups(
        transcript_names, transcript_to_gene,
        min_isoforms=args.min_isoforms)
    logger.info(
        f"Testing DTU for {len(gene_groups)} genes "
        f"across {len(valid_clusters)} clusters.")

    # Access the count matrix
    X = adata.X

    # One-vs-rest DTU testing
    dtu_results = []
    switching_events = []
    clusters = sorted(valid_clusters)

    for cluster in clusters:
        cluster_mask = np.array(adata.obs['cluster'] == cluster)
        rest_mask = ~cluster_mask

        n_cluster = cluster_mask.sum()
        n_rest = rest_mask.sum()
        if n_cluster == 0 or n_rest == 0:
            continue

        # Pre-slice matrix by cluster ONCE (avoids re-slicing per gene)
        if scipy.sparse.issparse(X):
            X_cluster = X[cluster_mask]
            X_rest = X[rest_mask]
        else:
            cluster_idx = np.where(cluster_mask)[0]
            rest_idx = np.where(rest_mask)[0]

        for gene_id, tx_indices in gene_groups.items():
            # Extract gene-level slice (small, safe to densify)
            tx_idx = np.array(tx_indices)

            if scipy.sparse.issparse(X):
                gene_counts_cluster = np.asarray(
                    X_cluster[:, tx_idx].toarray())
                gene_counts_rest = np.asarray(
                    X_rest[:, tx_idx].toarray())
            else:
                gene_counts_cluster = np.asarray(
                    X[np.ix_(cluster_idx, tx_idx)])
                gene_counts_rest = np.asarray(
                    X[np.ix_(rest_idx, tx_idx)])

            # Aggregated counts per transcript
            agg_cluster = gene_counts_cluster.sum(axis=0)
            agg_rest = gene_counts_rest.sum(axis=0)

            total_counts = agg_cluster.sum() + agg_rest.sum()
            if total_counts < args.min_gene_counts:
                continue

            # Count expressed isoforms
            expressed = ((agg_cluster + agg_rest) > 0).sum()
            if expressed < args.min_isoforms:
                continue

            # Run DTU test
            if args.test_method == 'dirichlet_multinomial':
                stat, pvalue, effect = dirichlet_multinomial_test(
                    gene_counts_cluster, gene_counts_rest)
            else:
                stat, pvalue, effect = chi_squared_dtu_test(
                    agg_cluster, agg_rest)

            gene_tx_names = [transcript_names[i] for i in tx_idx]
            dtu_results.append({
                'gene': gene_id,
                'cluster_a': cluster,
                'cluster_b': 'rest',
                'test_statistic': stat,
                'pvalue': pvalue,
                'n_transcripts': int(expressed),
                'effect_size': effect,
            })

            # Check for isoform switching
            switch = detect_isoform_switching(
                agg_cluster, agg_rest, gene_tx_names)
            if switch is not None:
                switch['gene'] = gene_id
                switch['cluster_a'] = cluster
                switch['cluster_b'] = 'rest'
                switching_events.append(switch)

    # Correct p-values
    if dtu_results:
        dtu_df = pd.DataFrame(dtu_results)
        pvals = dtu_df['pvalue'].values
        dtu_df['pvalue_adj'] = correct_pvalues(pvals)

        # Reorder columns
        dtu_df = dtu_df[[
            'gene', 'cluster_a', 'cluster_b', 'test_statistic',
            'pvalue', 'pvalue_adj', 'n_transcripts', 'effect_size'
        ]]

        # Sort by adjusted p-value
        dtu_df = dtu_df.sort_values('pvalue_adj').reset_index(drop=True)
    else:
        dtu_df = pd.DataFrame(columns=[
            'gene', 'cluster_a', 'cluster_b', 'test_statistic',
            'pvalue', 'pvalue_adj', 'n_transcripts', 'effect_size'
        ])

    # Write DTU results
    dtu_df.to_csv(args.output_dtu, sep='\t', index=False)
    logger.info(f"Wrote {len(dtu_df)} DTU test results to {args.output_dtu}.")

    # Write switching events
    if switching_events:
        switch_df = pd.DataFrame(switching_events)
        switch_df = switch_df[[
            'gene', 'cluster_a', 'cluster_b', 'dominant_transcript_a',
            'proportion_a', 'dominant_transcript_b', 'proportion_b',
            'switching_score'
        ]]
        switch_df = switch_df.sort_values(
            'switching_score', ascending=False).reset_index(drop=True)
    else:
        switch_df = pd.DataFrame(columns=[
            'gene', 'cluster_a', 'cluster_b', 'dominant_transcript_a',
            'proportion_a', 'dominant_transcript_b', 'proportion_b',
            'switching_score'
        ])
    switch_df.to_csv(args.output_switching, sep='\t', index=False)
    logger.info(
        f"Detected {len(switch_df)} isoform switching events.")

    # Summary statistics
    n_significant = 0
    top_genes = []
    if len(dtu_df) > 0:
        sig_mask = dtu_df['pvalue_adj'] <= args.fdr_threshold
        n_significant = int(sig_mask.sum())
        top_genes = dtu_df.loc[
            sig_mask, 'gene'].drop_duplicates().head(20).tolist()

    summary = {
        'n_genes_tested': int(
            dtu_df['gene'].nunique()) if len(dtu_df) > 0 else 0,
        'n_significant': n_significant,
        'n_switching_events': len(switch_df),
        'top_dtu_genes': top_genes,
    }
    with open(args.output_summary, 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        f"DTU analysis complete: {summary['n_genes_tested']} genes tested, "
        f"{n_significant} significant at FDR {args.fdr_threshold}, "
        f"{len(switch_df)} isoform switching events.")
