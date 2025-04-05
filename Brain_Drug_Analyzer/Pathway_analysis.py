import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from typing import Dict, List, Set, Optional, Union, Tuple

class PathwayAnalyzer:
    """
    A class for analyzing pathways associated with drug interactions in specific brain regions,
    including suppression/addiction ratio analysis.
    """
    
    def __init__(self, enrichment_file_path=None, output_dir="results"):
        """
        Initialize the PathwayAnalyzer with an optional enrichment file.
        
        Parameters:
        -----------
        enrichment_file_path : str, optional
            Path to the enrichment analysis file containing pathway information
        output_dir : str, default="results"
            Directory to save analysis outputs
        """
        self.pathway_to_genes = {}
        self.gene_to_pathways = defaultdict(list)
        self.significant_pval = 0.05
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
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
            
        Returns:
        --------
        bool
            True if data was loaded successfully, False otherwise
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
    
    def process_pathway_reports(self, 
                              addictive_dir: str, 
                              nonaddictive_dir: str, 
                              output_subdir: str = "ratio_analysis") -> Tuple[Dict, Dict]:
        """
        Process pathway report files to extract suppression/addiction ratios.
        
        Parameters:
        -----------
        addictive_dir : str
            Directory containing pathway reports for addictive drugs
        nonaddictive_dir : str
            Directory containing pathway reports for non-addictive drugs
        output_subdir : str, default="ratio_analysis"
            Subdirectory in output_dir to save results
            
        Returns:
        --------
        Tuple[Dict, Dict]
            Dictionaries with ratios for addictive and non-addictive drugs
        """
        # Create output subdirectory
        output_dir = os.path.join(self.output_dir, output_subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Process addictive drugs
        print("Processing addictive drug reports...")
        addictive_data = self._extract_ratios_from_reports(addictive_dir)
        
        # Process non-addictive drugs
        print("Processing non-addictive drug reports...")
        nonaddictive_data = self._extract_ratios_from_reports(nonaddictive_dir)
        
        # Generate heatmaps
        self.generate_ratio_heatmap(addictive_data, "Addictive", output_dir)
        self.generate_ratio_heatmap(nonaddictive_data, "Non-Addictive", output_dir)
        
        # Save data to CSV
        self._save_ratio_data_to_csv(addictive_data, nonaddictive_data, output_dir)
        
        # Create summary report
        self._create_ratio_analysis_summary(addictive_data, nonaddictive_data, addictive_dir, nonaddictive_dir, output_dir)
        
        return addictive_data, nonaddictive_data
    
    def _extract_ratios_from_reports(self, directory: str) -> Dict:
        """
        Extract suppression/addiction ratios from pathway report files.
        
        Parameters:
        -----------
        directory : str
            Directory containing pathway report files
            
        Returns:
        --------
        Dict
            Dictionary with structure {drug_name: {brain_region: ratio}}
        """
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return {}
        
        # Data structure to store results
        drug_region_ratios = defaultdict(dict)
        
        # Get all report files
        report_files = [f for f in os.listdir(directory) if f.endswith('_pathways.txt') 
                       and not f.startswith('Addiction_Genes_') 
                       and not f.startswith('Suppression_Genes_')]
        
        # Process each report file
        for report_file in report_files:
            # Use regex to extract drug name and brain region from filename
            match = re.match(r"(.+?)_(.+)_pathways\.txt", report_file)
            if match:
                drug_name = match.group(1)
                brain_region = match.group(2)
                
                file_path = os.path.join(directory, report_file)
                
                try:
                    # Extract combined gene counts and calculate ratio
                    total_addiction, total_suppression = self._extract_gene_counts_from_report(file_path)
                    
                    # Calculate ratio (suppression to addiction)
                    if total_addiction > 0:
                        ratio = total_suppression / total_addiction
                    else:
                        ratio = float('inf') if total_suppression > 0 else 0
                        
                    # Store the ratio
                    drug_region_ratios[drug_name][brain_region] = ratio
                        
                except Exception as e:
                    print(f"Error processing {report_file}: {e}")
        
        return drug_region_ratios
    
    def _extract_gene_counts_from_report(self, file_path: str) -> Tuple[int, int]:
        """
        Extract addiction and suppression gene counts from a drug pathway report file.
        
        Parameters:
        -----------
        file_path : str
            Path to the report file
            
        Returns:
        --------
        Tuple[int, int]
            (total_addiction_count, total_suppression_count)
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract normal and high expression gene counts
            normal_match = re.search(r"Normal Expression - Addiction Genes: (\d+) genes\nNormal Expression - Suppression Genes: (\d+) genes", content)
            high_match = re.search(r"High Expression - Addiction Genes: (\d+) genes\nHigh Expression - Suppression Genes: (\d+) genes", content)
            
            # Default to 0 if not found
            normal_addiction = int(normal_match.group(1)) if normal_match else 0
            normal_suppression = int(normal_match.group(2)) if normal_match else 0
            high_addiction = int(high_match.group(1)) if high_match else 0
            high_suppression = int(high_match.group(2)) if high_match else 0
            
            # Combine normal and high expression counts
            total_addiction = normal_addiction + high_addiction
            total_suppression = normal_suppression + high_suppression
            
            return total_addiction, total_suppression
                
        except Exception as e:
            print(f"Error extracting data from {file_path}: {e}")
            return 0, 0
    
    def generate_ratio_heatmap(self, 
                              data: Dict, 
                              drug_type: str, 
                              output_dir: str) -> None:
        """
        Generate a heatmap visualization of suppression/addiction ratios.
        
        Parameters:
        -----------
        data : Dict
            Dictionary with structure {drug_name: {brain_region: ratio}}
        drug_type : str
            Label for the drug type (e.g., "Addictive", "Non-Addictive")
        output_dir : str
            Directory to save the output file
        """
        # Convert to DataFrame for visualization
        rows = []
        
        for drug, regions in data.items():
            for region, ratio in regions.items():
                rows.append({
                    'Drug': drug,
                    'Region': region,
                    'Ratio': ratio if ratio != float('inf') else np.nan  # Handle infinity
                })
        
        if not rows:
            print(f"No data available to create heatmap for {drug_type} drugs")
            return
            
        df = pd.DataFrame(rows)
        
        # Create a pivot table for the heatmap
        pivot_df = df.pivot_table(
            values='Ratio', 
            index='Drug',
            columns='Region',
            aggfunc='mean'
        )
        
        # Sort drugs and regions alphabetically
        pivot_df = pivot_df.sort_index(axis=0)
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Determine sensible color scale values
        values = [v for v in df['Ratio'] if v != float('inf') and not np.isnan(v)]
        if values:
            vmin = max(1, np.percentile(values, 5))  # Use 5th percentile, but at least 1
            vmax = min(np.percentile(values, 95), 30)  # Use 95th percentile, cap at 30
        else:
            vmin = 1
            vmax = 20
        
        # Set up the figure
        fig_width = max(16, len(pivot_df.columns) * 0.5)  # Adjust width based on columns
        fig_height = max(8, len(pivot_df) * 0.5)  # Adjust height based on rows
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Generate heatmap
        ax = sns.heatmap(
            pivot_df, 
            cmap='Blues_r',  # Reverse blue colormap (darker = lower values)
            annot=True,      # Show values
            fmt='.1f',       # Format to 1 decimal place
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Suppression/Addiction Ratio'}
        )
        
        # Customize the plot
        plt.title(f'Suppression/Addiction Ratio for {drug_type} Drugs', fontsize=16)
        
        # Format x-axis labels - replace underscores with spaces
        x_labels = [label.replace('_', ' ') for label in pivot_df.columns]
        ax.set_xticklabels(x_labels, rotation=90, ha='center')
        
        # Format y-axis labels - replace underscores with spaces
        y_labels = [label.replace('_', ' ') for label in pivot_df.index]
        ax.set_yticklabels(y_labels, rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(output_dir, f'suppression_addiction_ratio_{drug_type.lower().replace("-", "_")}_drugs.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {drug_type} drugs heatmap to {filename}")
    
    def _save_ratio_data_to_csv(self, 
                              addictive_data: Dict, 
                              nonaddictive_data: Dict, 
                              output_dir: str) -> None:
        """
        Save ratio data to CSV files for further analysis.
        
        Parameters:
        -----------
        addictive_data : Dict
            Data for addictive drugs
        nonaddictive_data : Dict
            Data for non-addictive drugs
        output_dir : str
            Directory to save output files
        """
        # Convert addictive drug data to DataFrame
        addictive_rows = []
        for drug, regions in addictive_data.items():
            for region, ratio in regions.items():
                addictive_rows.append({
                    'Drug': drug,
                    'Region': region,
                    'Ratio': ratio if ratio != float('inf') else 'Inf',
                    'Type': 'Addictive'
                })
        
        # Convert non-addictive drug data to DataFrame
        nonaddictive_rows = []
        for drug, regions in nonaddictive_data.items():
            for region, ratio in regions.items():
                nonaddictive_rows.append({
                    'Drug': drug,
                    'Region': region,
                    'Ratio': ratio if ratio != float('inf') else 'Inf',
                    'Type': 'Non-Addictive'
                })
        
        # Combine both datasets
        all_rows = addictive_rows + nonaddictive_rows
        combined_df = pd.DataFrame(all_rows)
        
        # Save to CSV files
        combined_df.to_csv(os.path.join(output_dir, 'all_suppression_addiction_ratios.csv'), index=False)
        
        # Create pivot tables for addictive and non-addictive drugs
        if addictive_rows:
            addictive_df = pd.DataFrame(addictive_rows)
            addictive_pivot = addictive_df.pivot_table(
                values='Ratio', 
                index='Drug',
                columns='Region',
                aggfunc='mean'
            )
            addictive_pivot.to_csv(os.path.join(output_dir, 'addictive_drugs_ratio_matrix.csv'))
        
        if nonaddictive_rows:
            nonaddictive_df = pd.DataFrame(nonaddictive_rows)
            nonaddictive_pivot = nonaddictive_df.pivot_table(
                values='Ratio', 
                index='Drug',
                columns='Region',
                aggfunc='mean'
            )
            nonaddictive_pivot.to_csv(os.path.join(output_dir, 'nonaddictive_drugs_ratio_matrix.csv'))
        
        print(f"Saved ratio data to CSV files in {output_dir}")
    
    def _create_ratio_analysis_summary(self,
                                     addictive_data: Dict,
                                     nonaddictive_data: Dict,
                                     addictive_dir: str,
                                     nonaddictive_dir: str,
                                     output_dir: str) -> None:
        """
        Create a summary report of the ratio analysis.
        
        Parameters:
        -----------
        addictive_data : Dict
            Data for addictive drugs
        nonaddictive_data : Dict
            Data for non-addictive drugs
        addictive_dir : str
            Directory containing addictive drug reports
        nonaddictive_dir : str
            Directory containing non-addictive drug reports
        output_dir : str
            Directory to save output files
        """
        # Extract values for summary statistics
        addictive_values = []
        for drug, regions in addictive_data.items():
            for region, ratio in regions.items():
                if ratio != float('inf') and not np.isnan(ratio):
                    addictive_values.append(ratio)
        
        nonaddictive_values = []
        for drug, regions in nonaddictive_data.items():
            for region, ratio in regions.items():
                if ratio != float('inf') and not np.isnan(ratio):
                    nonaddictive_values.append(ratio)
        
        # Save summary to file
        summary_file = os.path.join(output_dir, "ratio_analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("Suppression/Addiction Ratio Analysis Summary\n")
            f.write("==========================================\n\n")
            f.write(f"Addictive Drugs Directory: {addictive_dir}\n")
            f.write(f"Non-Addictive Drugs Directory: {nonaddictive_dir}\n\n")
            f.write(f"Number of Addictive Drugs Analyzed: {len(addictive_data)}\n")
            f.write(f"Number of Non-Addictive Drugs Analyzed: {len(nonaddictive_data)}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write("-----------------\n\n")
            
            if addictive_values:
                f.write(f"Addictive Drugs (n={len(addictive_values)}):\n")
                f.write(f"  Mean Ratio: {np.mean(addictive_values):.2f}\n")
                f.write(f"  Median Ratio: {np.median(addictive_values):.2f}\n")
                f.write(f"  Standard Deviation: {np.std(addictive_values):.2f}\n")
                f.write(f"  Range: {min(addictive_values):.2f} - {max(addictive_values):.2f}\n\n")
            else:
                f.write("No data available for addictive drugs\n\n")
                
            if nonaddictive_values:
                f.write(f"Non-addictive Drugs (n={len(nonaddictive_values)}):\n")
                f.write(f"  Mean Ratio: {np.mean(nonaddictive_values):.2f}\n")
                f.write(f"  Median Ratio: {np.median(nonaddictive_values):.2f}\n")
                f.write(f"  Standard Deviation: {np.std(nonaddictive_values):.2f}\n")
                f.write(f"  Range: {min(nonaddictive_values):.2f} - {max(nonaddictive_values):.2f}\n\n")
            else:
                f.write("No data available for non-addictive drugs\n\n")
            
            f.write("Files Generated:\n")
            f.write("--------------\n")
            f.write("- Heatmaps for addictive and non-addictive drugs\n")
            f.write("- CSV files with ratio data for further analysis\n")
        
        print(f"Saved analysis summary to {summary_file}")
    
    def create_pathway_report(self, 
                            drug_name: str, 
                            region_name: str, 
                            analysis_results: Dict, 
                            output_path: str = None) -> str:
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
        # Calculate suppression/addiction ratio
        normal_addiction = len(analysis_results['normal_expression']['addiction_genes'])
        normal_suppression = len(analysis_results['normal_expression']['suppression_genes'])
        high_addiction = len(analysis_results['high_expression']['addiction_genes'])
        high_suppression = len(analysis_results['high_expression']['suppression_genes'])
        
        total_addiction = normal_addiction + high_addiction
        total_suppression = normal_suppression + high_suppression
        
        if total_addiction > 0:
            ratio = total_suppression / total_addiction
        else:
            ratio = float('inf') if total_suppression > 0 else 0
        
        # Build the report
        report = []
        report.append(f"Pathway Analysis for {drug_name} in {region_name}")
        report.append("=" * 80 + "\n")
        
        # Addiction and Suppression Gene Summary
        report.append("ADDICTION AND SUPPRESSION GENE SUMMARY")
        report.append("-" * 80)
        report.append(f"Normal Expression - Addiction Genes: {normal_addiction} genes")
        report.append(f"Normal Expression - Suppression Genes: {normal_suppression} genes")
        report.append(f"High Expression - Addiction Genes: {high_addiction} genes")
        report.append(f"High Expression - Suppression Genes: {high_suppression} genes\n")
        
        report.append(f"Total Addiction Genes: {total_addiction}")
        report.append(f"Total Suppression Genes: {total_suppression}")
        report.append(f"Suppression/Addiction Ratio: {ratio:.2f}\n")
        
        # List specific genes
        normal_addiction_genes = analysis_results['normal_expression']['addiction_genes']
        if normal_addiction_genes:
            report.append(f"Addiction genes with normal expression: {', '.join(sorted(normal_addiction_genes))}")
            
        normal_suppression_genes = analysis_results['normal_expression']['suppression_genes']
        if normal_suppression_genes:
            report.append(f"Suppression genes with normal expression: {', '.join(sorted(normal_suppression_genes))}")
            
        high_addiction_genes = analysis_results['high_expression']['addiction_genes']
        if high_addiction_genes:
            report.append(f"Addiction genes with high expression: {', '.join(sorted(high_addiction_genes))}")
            
        high_suppression_genes = analysis_results['high_expression']['suppression_genes']
        if high_suppression_genes:
            report.append(f"Suppression genes with high expression: {', '.join(sorted(high_suppression_genes))}\n")
        
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
                    if gene in normal_addiction_genes:
                        categories.append("A")
                        addiction_count += 1
                    if gene in normal_suppression_genes:
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
                    if gene in high_addiction_genes:
                        categories.append("A")
                        addiction_count += 1
                    if gene in high_suppression_genes:
                        categories.append("S")
                        suppression_count += 1
                    
                    if categories:
                        formatted_genes.append(f"{gene} [{','.join(categories)}]")
                    else:
                        formatted_genes.append(gene)
                
                report.append(f"Drug genes in pathway ({len(genes)}): {', '.join(formatted_genes)}")
                report.append(f"Addiction genes: {addiction_count}, Suppression genes: {suppression_count}")
                
                # Calculate pathway coverage
                if pathway in self.pathway_to
