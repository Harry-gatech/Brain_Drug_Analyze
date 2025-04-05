import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Set, Optional, Union, Tuple

class PathwayAnalyzer:
    """
    A class for analyzing pathways associated with drug interactions in specific brain regions.
    This component integrates with the network analysis to provide pathway enrichment analysis.
    """
    
    def __init__(self, enrichment_file_path=None):
        """
        Initialize the PathwayAnalyzer with an optional enrichment file.
        
        Parameters:
        -----------
        enrichment_file_path : str, optional
            Path to the enrichment analysis file containing pathway information
        """
        self.pathway_to_genes = {}
        self.gene_to_pathways = defaultdict(list)
        self.significant_pval = 0.05
        
        if enrichment_file_path and os.path.exists(enrichment_file_path):
            self.load_enrichment_data(enrichment_file_path)
    
    def load_enrichment_data(self, enrichment_file_path, separator='\t'):
        """
        Load pathway enrichment data from a file.
        
        Parameters:
        -----------
        enrichment_file_path : str
            Path to the enrichment analysis file
        separator : str, default='\t'
            Separator used in the enrichment file
        """
        try:
            df_enrichment = pd.read_csv(enrichment_file_path, sep=separator, on_bad_lines='skip')
            
            # Identify column names
            p_value_cols = [col for col in df_enrichment.columns if 'adjust' in col.lower() and 'p' in col.lower()]
            pval_column = p_value_cols[0] if p_value_cols else 'Adjusted P-value'
            
            gene_column = 'Genes'
            if 'Genes' not in df_enrichment.columns:
                gene_cols = [col for col in df_enrichment.columns if 'gene' in col.lower()]
                gene_column = gene_cols[0] if gene_cols else 'Genes'
            
            # Filter for significant pathways
            df_filtered = df_enrichment[df_enrichment[pval_column] <= self.significant_pval]
            
            # Create mapping dictionaries
            term_column = 'Term' if 'Term' in df_filtered.columns else df_filtered.columns[0]
            
            for _, row in df_filtered.iterrows():
                pathway = row[term_column]
                if pd.notna(row[gene_column]) and isinstance(row[gene_column], str):
                    genes = [gene.strip() for gene in row[gene_column].split(';')]
                    self.pathway_to_genes[pathway] = genes
                    
                    # Create gene to pathway mapping
                    for gene in genes:
                        self.gene_to_pathways[gene].append(pathway)
            
            print(f"Loaded {len(self.pathway_to_genes)} significant pathways from enrichment data")
            return True
            
        except Exception as e:
            print(f"Error loading enrichment data: {e}")
            return False
    
    def analyze_drug_region_pathways(self, 
                                    drug_genes: Set[str], 
                                    region_genes: Dict[str, Set[str]], 
                                    addiction_genes: Set[str], 
                                    suppression_genes: Set[str]) -> Dict:
        """
        Analyze pathways for a drug in a specific brain region.
        
        Parameters:
        -----------
        drug_genes : Set[str]
            Set of genes targeted by the drug
        region_genes : Dict[str, Set[str]]
            Dictionary with 'normal' and 'high' expression genes in the region
        addiction_genes : Set[str]
            Set of genes associated with addiction
        suppression_genes : Set[str]
            Set of genes associated with addiction suppression
            
        Returns:
        --------
        Dict
            Analysis results containing pathway information
        """
        results = {
            'normal_expression': {
                'all_genes': set(),
                'addiction_genes': set(),
                'suppression_genes': set(),
                'pathways': defaultdict(set)
            },
            'high_expression': {
                'all_genes': set(),
                'addiction_genes': set(),
                'suppression_genes': set(),
                'pathways': defaultdict(set)
            }
        }
        
        # Process normal expression genes
        normal_genes = region_genes.get('normal', set())
        normal_drug_genes = normal_genes.intersection(drug_genes)
        results['normal_expression']['all_genes'] = normal_drug_genes
        results['normal_expression']['addiction_genes'] = normal_drug_genes.intersection(addiction_genes)
        results['normal_expression']['suppression_genes'] = normal_drug_genes.intersection(suppression_genes)
        
        # Get pathways for normal expression
        for gene in normal_drug_genes:
            if gene in self.gene_to_pathways:
                for pathway in self.gene_to_pathways[gene]:
                    results['normal_expression']['pathways'][pathway].add(gene)
        
        # Process high expression genes
        high_genes = region_genes.get('high', set())
        high_drug_genes = high_genes.intersection(drug_genes)
        results['high_expression']['all_genes'] = high_drug_genes
        results['high_expression']['addiction_genes'] = high_drug_genes.intersection(addiction_genes)
        results['high_expression']['suppression_genes'] = high_drug_genes.intersection(suppression_genes)
        
        # Get pathways for high expression
        for gene in high_drug_genes:
            if gene in self.gene_to_pathways:
                for pathway in self.gene_to_pathways[gene]:
                    results['high_expression']['pathways'][pathway].add(gene)
        
        return results
    
    def calculate_suppression_addiction_ratio(self, 
                                             drug_genes: Set[str], 
                                             region_genes: Dict[str, Set[str]], 
                                             addiction_genes: Set[str], 
                                             suppression_genes: Set[str]) -> float:
        """
        Calculate the suppression/addiction ratio for a drug in a specific brain region.
        
        Parameters:
        -----------
        drug_genes : Set[str]
            Set of genes targeted by the drug
        region_genes : Dict[str, Set[str]]
            Dictionary with 'normal' and 'high' expression genes in the region
        addiction_genes : Set[str]
            Set of genes associated with addiction
        suppression_genes : Set[str]
            Set of genes associated with addiction suppression
            
        Returns:
        --------
        float
            Suppression/addiction ratio
        """
        # Combine normal and high expression genes
        all_region_genes = region_genes.get('normal', set()).union(region_genes.get('high', set()))
        
        # Find drug genes in this region
        drug_region_genes = all_region_genes.intersection(drug_genes)
        
        # Count addiction and suppression genes
        addiction_count = len(drug_region_genes.intersection(addiction_genes))
        suppression_count = len(drug_region_genes.intersection(suppression_genes))
        
        # Calculate ratio (avoid division by zero)
        if addiction_count == 0:
            return float('inf')  # Return infinity if no addiction genes
        
        return suppression_count / addiction_count
    
    def generate_ratio_heatmap(self, 
                              ratio_data: Dict[str, Dict[str, float]], 
                              output_path: str = None,
                              title: str = "Suppression/Addiction Ratio Heatmap"):
        """
        Generate a heatmap visualization of suppression/addiction ratios across drugs and brain regions.
        
        Parameters:
        -----------
        ratio_data : Dict[str, Dict[str, float]]
            Nested dictionary with format {drug_name: {region_name: ratio}}
        output_path : str, optional
            Path to save the heatmap image
        title : str, default="Suppression/Addiction Ratio Heatmap"
            Title for the heatmap
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object with the heatmap
        """
        # Convert to DataFrame
        drugs = list(ratio_data.keys())
        all_regions = set()
        for drug_ratios in ratio_data.values():
            all_regions.update(drug_ratios.keys())
        
        regions = sorted(all_regions)
        
        # Create matrix for heatmap
        matrix = np.zeros((len(drugs), len(regions)))
        
        for i, drug in enumerate(drugs):
            for j, region in enumerate(regions):
                if region in ratio_data[drug]:
                    matrix[i, j] = ratio_data[drug][region]
        
        # Create figure and heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis",
                         xticklabels=regions, yticklabels=drugs)
        plt.title(title)
        plt.xlabel("Brain Regions")
        plt.ylabel("Drugs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return plt.gcf()
    
    def create_pathway_report(self, 
                             drug_name: str, 
                             region_name: str, 
                             analysis_results: Dict, 
                             output_path: str = None):
        """
        Create a detailed report of pathway analysis for a drug in a brain region.
        
        Parameters:
        -----------
        drug_name : str
            Name of the drug
        region_name : str
            Name of the brain region
        analysis_results : Dict
            Results from analyze_drug_region_pathways function
        output_path : str, optional
            Path to save the report file
            
        Returns:
        --------
        str
            Report content as a string
        """
        report = []
        report.append(f"Pathway Analysis for {drug_name} in {region_name}")
        report.append("=" * 80 + "\n")
        
        # Addiction and Suppression Gene Summary
        report.append("ADDICTION AND SUPPRESSION GENE SUMMARY")
        report.append("-" * 80)
        
        normal_addiction = analysis_results['normal_expression']['addiction_genes']
        normal_suppression = analysis_results['normal_expression']['suppression_genes']
        high_addiction = analysis_results['high_expression']['addiction_genes']
        high_suppression = analysis_results['high_expression']['suppression_genes']
        
        report.append(f"Normal Expression - Addiction Genes: {len(normal_addiction)} genes")
        report.append(f"Normal Expression - Suppression Genes: {len(normal_suppression)} genes")
        report.append(f"High Expression - Addiction Genes: {len(high_addiction)} genes")
        report.append(f"High Expression - Suppression Genes: {len(high_suppression)} genes\n")
        
        if normal_addiction:
            report.append(f"Addiction genes with normal expression: {', '.join(sorted(normal_addiction))}")
        if normal_suppression:
            report.append(f"Suppression genes with normal expression: {', '.join(sorted(normal_suppression))}")
        if high_addiction:
            report.append(f"Addiction genes with high expression: {', '.join(sorted(high_addiction))}")
        if high_suppression:
            report.append(f"Suppression genes with high expression: {', '.join(sorted(high_suppression))}\n")
        
        # Normal expression pathways
        report.append("NORMAL EXPRESSION PATHWAYS")
        report.append("-" * 80)
        normal_genes = analysis_results['normal_expression']['all_genes']
        normal_pathways = analysis_results['normal_expression']['pathways']
        
        report.append(f"Found {len(normal_genes)} {drug_name} genes with normal expression")
        report.append(f"These genes are associated with {len(normal_pathways)} pathways\n")
        
        if normal_pathways:
            # Sort pathways by number of genes (descending)
            sorted_pathways = sorted(normal_pathways.items(), key=lambda x: len(x[1]), reverse=True)
            
            for pathway, genes in sorted_pathways:
                report.append(f"Pathway: {pathway}")
                
                # Format genes with categories
                formatted_genes = []
                addiction_count = 0
                suppression_count = 0
                
                for gene in sorted(genes):
                    categories = []
                    if gene in normal_addiction:
                        categories.append("A")
                        addiction_count += 1
                    if gene in normal_suppression:
                        categories.append("S")
                        suppression_count += 1
                    
                    if categories:
                        formatted_genes.append(f"{gene} [{','.join(categories)}]")
                    else:
                        formatted_genes.append(gene)
                
                report.append(f"Drug genes in pathway ({len(genes)}): {', '.join(formatted_genes)}")
                report.append(f"Addiction genes: {addiction_count}, Suppression genes: {suppression_count}")
                
                # Calculate pathway coverage
                if pathway in self.pathway_to_genes:
                    all_pathway_genes = set(self.pathway_to_genes[pathway])
                    coverage_percent = (len(genes) / len(all_pathway_genes)) * 100 if all_pathway_genes else 0
                    report.append(f"Pathway coverage: {len(genes)}/{len(all_pathway_genes)} genes ({coverage_percent:.2f}%)\n")
                else:
                    report.append("")
        else:
            report.append("No pathways found for normally expressed drug genes\n")
        
        # High expression pathways
        report.append("\nHIGH EXPRESSION PATHWAYS")
        report.append("-" * 80)
        high_genes = analysis_results['high_expression']['all_genes']
        high_pathways = analysis_results['high_expression']['pathways']
        
        report.append(f"Found {len(high_genes)} {drug_name} genes with high expression")
        report.append(f"These genes are associated with {len(high_pathways)} pathways\n")
        
        if high_pathways:
            # Sort pathways by number of genes (descending)
            sorted_pathways = sorted(high_pathways.items(), key=lambda x: len(x[1]), reverse=True)
            
            for pathway, genes in sorted_pathways:
                report.append(f"Pathway: {pathway}")
                
                # Format genes with categories
                formatted_genes = []
                addiction_count = 0
                suppression_count = 0
                
                for gene in sorted(genes):
                    categories = []
                    if gene in high_addiction:
                        categories.append("A")
                        addiction_count += 1
                    if gene in high_suppression:
                        categories.append("S")
                        suppression_count += 1
                    
                    if categories:
                        formatted_genes.append(f"{gene} [{','.join(categories)}]")
                    else:
                        formatted_genes.append(gene)
                
                report.append(f"Drug genes in pathway ({len(genes)}): {', '.join(formatted_genes)}")
                report.append(f"Addiction genes: {addiction_count}, Suppression genes: {suppression_count}")
                
                # Calculate pathway coverage
                if pathway in self.pathway_to_genes:
                    all_pathway_genes = set(self.pathway_to_genes[pathway])
                    coverage_percent = (len(genes) / len(all_pathway_genes)) * 100 if all_pathway_genes else 0
                    report.append(f"Pathway coverage: {len(genes)}/{len(all_pathway_genes)} genes ({coverage_percent:.2f}%)\n")
                else:
                    report.append("")
        else:
            report.append("No pathways found for highly expressed drug genes\n")
        
        # Save report if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(report))
        
        return '\n'.join(report)
