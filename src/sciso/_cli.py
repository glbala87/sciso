"""sciso command-line interface."""
import argparse
import importlib
import sys

from . import __version__
from ._logging import get_main_logger

MODULES = {
    'cluster': ('sciso.cluster_analysis', 'Basic Scanpy clustering'),
    'dual-cluster': (
        'sciso.dual_layer_clustering',
        'Gene + isoform dual-layer clustering'),
    'dtu': (
        'sciso.differential_transcript_usage',
        'Differential transcript usage testing'),
    'novel-isoforms': (
        'sciso.novel_isoform_discovery',
        'Cluster-specific novel isoform discovery'),
    'trajectory': (
        'sciso.isoform_trajectory',
        'Isoform-aware trajectory analysis'),
    'ase': (
        'sciso.allele_specific_expression',
        'Allele-specific expression analysis'),
    'annotate': (
        'sciso.cell_type_annotation',
        'Cell type annotation'),
    'export': (
        'sciso.export_anndata',
        'Export unified AnnData h5ad'),
    'benchmark-dtu': (
        'sciso.benchmark_dtu',
        'DTU detection benchmarking'),
    'report': (
        'sciso.report',
        'Generate HTML report'),
    'plot': (
        'sciso.plot',
        'Generate publication-quality figures'),
    'compare': (
        'sciso.multi_sample',
        'Cross-sample comparison of sciso results'),
    'validate': (
        'sciso.validate',
        'Validate pipeline inputs'),
    'run': (
        'sciso.pipeline',
        'Run full sciso pipeline'),
}


def cli():
    """sciso CLI entry point."""
    parser = argparse.ArgumentParser(
        'sciso',
        description=(
            'sciso: single-cell isoform analysis. '
            'Isoform-aware scRNA-seq analysis for long-read data.'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--version', action='version',
        version=f'sciso {__version__}')

    subparsers = parser.add_subparsers(
        title='commands', dest='command',
        description='Available analysis modules:')

    # Register all modules
    for cmd_name, (module_path, description) in MODULES.items():
        try:
            mod = importlib.import_module(module_path)
            sub_parser = mod.argparser()
            subparsers.add_parser(
                cmd_name, parents=[sub_parser],
                description=description,
                help=description,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        except (ImportError, AttributeError):
            # Module not available or no argparser
            pass

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logger = get_main_logger("sciso")

    module_path = MODULES[args.command][0]
    mod = importlib.import_module(module_path)
    mod.main(args)
