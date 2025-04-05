"""
Brain Drug Analyzer

A Python module for analyzing drug interactions with addiction and suppression genes across brain regions.
Based on research in "Brain region specific gene signatures in addiction and addiction suppression".

This module provides tools for:
- Network analysis of drug-gene interactions
- Brain region-specific gene expression analysis
- Suppression/addiction gene ratio calculation
- Visualization of drug interaction patterns
"""

# Import network analysis components
from .Network_analysis import (
    fetch_string_interactions,
    fetch_stitch_interactions,
    fetch_dgidb_interactions,
    get_pubchem_cid,
    build_gene_network,
    build_drug_network,
    merge_networks,
    extract_drug_subnetwork,
    analyze_subnetwork_genes,
    visualize_subnetwork
)

# Import analyzer class
from .Analyzer import DrugBrainAnalyzer
from .Pathway_analysis import PathwayAnalyzer

# Define module-level constants
ADDICTION_BRAIN_REGIONS = [
    "Ventral_Tegmental_Area",
    "Nucleus_Accumbens",
    "Putamen",
    "Amygdala",
    "Hippocampus",
    "Habenula",
    "Anterior_Cingulate_Cortex",
    "Orbitofrontal_Gyrus",
    "Caudate_Nucleus",
    "Insular_Cortex"
]

# Module information
__version__ = '0.1.0'
__author__ = 'Brain Drug Analyzer Team'
__description__ = 'Tools for analyzing drug interactions in brain addiction networks'
