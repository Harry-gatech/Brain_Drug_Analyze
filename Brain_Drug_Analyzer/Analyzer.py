# Analyzer.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Optional, Union, Tuple
import networkx as nx

class DrugBrainAnalyzer:
    """
    A comprehensive analyzer for studying drug interactions with addiction and suppression genes
    across brain regions. This class serves as the main container and coordinator for the analysis
    workflow described in the paper "Brain region specific gene signatures in addiction and 
    addiction suppression".
    
    The analyzer loads and manages:
    - Addiction and suppression gene sets
    - Brain region expression data (categorized as low, normal, high)
    - Drug target data
    
    It coordinates:
    - Network analysis of drug-gene interactions
    - Pathway analysis in specific brain regions
    - Suppression/addiction ratio calculation and visualization
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the DrugBrainAnalyzer.
        
        Parameters:
        -----------
        output_dir : str, default="results"
            Directory to save analysis outputs
        """
        self.addiction_genes = set()
        self.suppression_genes = set()
        self.categorized_regions = {}  # Dictionary for mapping brain regions to categorized genes
        self.drugs = {}  # Drugs and their targets
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_gene_sets(self, addiction_gene_file: str, suppression_gene_file: str) -> None:
        """
        Load addiction and suppression gene sets from text files.
        
        Parameters:
        -----------
        addiction_gene_file : str
            Path to file containing addiction-associated genes
        suppression_gene_file : str
            Path to file containing suppression-associated genes
        """
        # Load addiction genes
        with open(addiction_gene_file, 'r') as f:
            self.addiction_genes = set(line.strip() for line in f if line.strip())

        # Load suppression genes
        with open(suppression_gene_file, 'r') as f:
            self.suppression_genes = set(line.strip() for line in f if line.strip())
        
        print(f"Loaded {len(self.addiction_genes)} addiction genes and {len(self.suppression_genes)} suppression genes")

    def load_categorized_brain_data(self, brain_directory: str) -> None:
        """
        Load categorized gene expression data from brain region directories.
        
        Parameters:
        -----------
        brain_directory : str
            Path to directory containing brain region subdirectories.
            Each brain region should have Low, Normal, and High subdirectories
            with gene expression data in CSV files.
        """
        brain_regions = [d for d in os.listdir(brain_directory) 
                         if os.path.isdir(os.path.join(brain_directory, d))]
        
        for region in brain_regions:
            region_path = os.path.join(brain_directory, region)

            low_genes = set()
            normal_genes = set()
            high_genes = set()

            # Load Low expression genes
            low_path = os.path.join(region_path, "Low")
            if os.path.exists(low_path):
                for file in os.listdir(low_path):
                    if file.endswith(".csv"):
                        try:
                            df = pd.read_csv(os.path.join(low_path, file))
                            gene_column = df.columns[0]
                            low_genes.update(df[gene_column].astype(str).tolist())
                        except Exception as e:
                            print(f"Error reading {file} in {low_path}: {e}")
            
            # Load Normal expression genes
            normal_path = os.path.join(region_path, "Normal")
            if os.path.exists(normal_path):
                for file in os.listdir(normal_path):
                    if file.endswith(".csv"):
                        try:
                            df = pd.read_csv(os.path.join(normal_path, file))
                            gene_column = df.columns[0]
                            normal_genes.update(df[gene_column].astype(str).tolist())
                        except Exception as e:
                            print(f"Warning: Error reading {file} in {normal_path}: {e}")
        
            # Load High expression genes
            high_path = os.path.join(region_path, "High")
            if os.path.exists(high_path):
                for file in os.listdir(high_path):
                    if file.endswith(".csv"):
                        try:
                            df = pd.read_csv(os.path.join(high_path, file))
                            gene_column = df.columns[0]
                            high_genes.update(df[gene_column].astype(str).tolist())
                        except Exception as e:
                            print(f"Warning: Error reading {file} in {high_path}: {e}")

            # Store the genes for this region
            self.categorized_regions[region] = {
                'low': low_genes,
                'normal': normal_genes,
                'high': high_genes
            }
        
        print(f"Loaded gene expression data for {len(self.categorized_regions)} brain regions")

    def load_drug_targets(self, drug_file_path: str, drug_name: Optional[str] = None) -> None:
        """
        Load target genes for a drug from a file.
        
        Parameters:
        -----------
        drug_file_path : str
            Path to file containing drug target genes
        drug_name : str, optional
            Name to use for the drug. If not provided, uses the filename.
        """
        if drug_name is None:
            drug_name = os.path.basename(drug_file_path).split('.')[0]
        
        try:
            # Load targets based on file type
            if drug_file_path.endswith('.csv'):
                df = pd.read_csv(drug_file_path)
                # Assume first column contains gene names
                targets = set(df.iloc[:, 0].astype(str).tolist())
            else:
                # Assume text file with one gene per line
                with open(drug_file_path, 'r') as f:
                    targets = set(line.strip() for line in f if line.strip())
            
            # Store targets
            self.drugs[drug_name] = targets
            print(f"Loaded {len(targets)} target genes for {drug_name}")
        
        except Exception as e:
            print(f"Error loading drug targets for {drug_name}: {e}")
    
    def load_drug_targets_from_directory(self, drug_dir: str) -> None:
        """
        Load drug targets from all files in a directory.
        
        Parameters:
        -----------
        drug_dir : str
            Path to directory containing drug target files
        """
        if not os.path.exists(drug_dir):
            print(f"Directory does not exist: {drug_dir}")
            return
        
        for drug_file in os.listdir(drug_dir):
            if drug_file.endswith('.txt') or drug_file.endswith('.csv'):
                drug_file_path = os.path.join(drug_dir, drug_file)
                self.load_drug_targets(drug_file_path)
        
        print(f"Loaded targets for {len(self.drugs)} drugs")

    def get_region_genes(self, region_name: str) -> Dict[str, Set[str]]:
        """
        Get genes for a specific brain region.
        
        Parameters:
        -----------
        region_name : str
            Name of the brain region
            
        Returns:
        --------
        Dict[str, Set[str]]
            Dictionary with 'low', 'normal', and 'high' expression genes
        """
        if region_name not in self.categorized_regions:
            print(f"Warning: Region '{region_name}' not found in brain data")
            return {'low': set(), 'normal': set(), 'high': set()}
        
        return self.categorized_regions[region_name]

    def analyze_drug_network(self, drug_name: str) -> Dict:
        """
        Perform network analysis for a drug to identify addiction and suppression
        gene interactions.
        
        Parameters:
        -----------
        drug_name : str
            Name of the drug to analyze
            
        Returns:
        --------
        Dict
            Analysis results including gene counts and ratio
        """
        from .Network_analysis import (
            build_gene_network, build_drug_network, merge_networks,
            extract_drug_subnetwork, analyze_subnetwork_genes
        )
        
        print(f"Analyzing network for {drug_name}...")
        
        # Build gene network
        gene_network = build_gene_network(
            self.addiction_genes, 
            self.suppression_genes, 
            confidence_score=0.4
        )
        
        # Build drug network
        drug_network = build_drug_network(drug_name, confidence_score=0.4)
        
        # Merge networks
        merged_network = merge_networks(gene_network, drug_network)
        
        # Extract subnetwork with drug interactions
        subnetwork = extract_drug_subnetwork(merged_network, drug_name, max_distance=2)
        
        # Analyze interaction patterns
        analysis = analyze_subnetwork_genes(
            subnetwork, 
            drug_name, 
            self.addiction_genes, 
            self.suppression_genes
        )
        
        # Save visualization if output directory is set
        if self.output_dir:
            from .Network_analysis import visualize_subnetwork
            fig = visualize_subnetwork(subnetwork)
            os.makedirs(self.output_dir, exist_ok=True)
            fig.savefig(os.path.join(self.output_dir, f"{drug_name}_network.png"))
            plt.close(fig)
        
        return analysis

    def analyze_drug_region_pathways(self, 
                                    drug_name: str, 
                                    region_name: str, 
                                    enrichment_file: Optional[str] = None,
                                    output_path: Optional[str] = None) -> Dict:
        """
        Analyze pathways for a drug in a specific brain region.
        
        Parameters:
        -----------
        drug_name : str
            Name of the drug to analyze
        region_name : str
            Name of the brain region to analyze
        enrichment_file : str, optional
            Path to enrichment file with pathway data
        output_path : str, optional
            Path to save the pathway report
            
        Returns:
        --------
        Dict
            Pathway analysis results
        """
        if drug_name not in self.drugs:
            raise ValueError(f"Drug {drug_name} not loaded. Use load_drug_targets() first.")
        
        if region_name not in self.categorized_regions:
            raise ValueError(f"Region {region_name} not loaded. Use load_categorized_brain_data() first.")
        
        from .Pathway_analysis import PathwayAnalyzer
        
        # Initialize pathway analyzer
        pathway_analyzer = PathwayAnalyzer(
            enrichment_file_path=enrichment_file,
            output_dir=self.output_dir
        )
        
        # Get region genes
        region_genes = {
            'normal': self.categorized_regions[region_name]['normal'],
            'high': self.categorized_regions[region_name]['high']
        }
        
        # Get drug genes
        drug_genes = self.drugs[drug_name]
        
        # Analyze pathways
        results = pathway_analyzer.analyze_drug_region_pathways(
            drug_genes,
            region_genes,
            self.addiction_genes,
            self.suppression_genes
        )
        
        # Calculate ratio
        ratio = pathway_analyzer.calculate_suppression_addiction_ratio(
            drug_genes,
            region_genes,
            self.addiction_genes,
            self.suppression_genes
        )
        
        # Add ratio to results
        results['suppression_addiction_ratio'] = ratio
        
        # Generate report if output path is provided
        if output_path or self.output_dir:
            if not output_path and self.output_dir:
                output_path = os.path.join(self.output_dir, f"{drug_name}_{region_name}_pathways.txt")
            
            pathway_analyzer.create_pathway_report(
                drug_name,
                region_name,
                results,
                output_path
            )
        
        return results

    def calculate_suppression_addiction_ratios(self, 
                                              drug_names: Optional[List[str]] = None,
                                              region_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate suppression/addiction ratios for multiple drugs across brain regions.
        
        Parameters:
        -----------
        drug_names : List[str], optional
            List of drug names to analyze. If None, uses all loaded drugs.
        region_names : List[str], optional
            List of brain regions to analyze. If None, uses all loaded regions.
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Nested dictionary with structure {drug_name: {region_name: ratio}}
        """
        from .Pathway_analysis import PathwayAnalyzer
        
        # Use all loaded drugs if not specified
        if drug_names is None:
            drug_names = list(self.drugs.keys())
        
        # Use all loaded regions if not specified
        if region_names is None:
            region_names = list(self.categorized_regions.keys())
        
        # Initialize pathway analyzer
        pathway_analyzer = PathwayAnalyzer(output_dir=self.output_dir)
        
        # Calculate ratios for each drug and region
        all_ratios = {}
        
        for drug_name in drug_names:
            if drug_name not in self.drugs:
                print(f"Warning: Drug {drug_name} not loaded, skipping...")
                continue
            
            all_ratios[drug_name] = {}
            drug_genes = self.drugs[drug_name]
            
            for region_name in region_names:
                if region_name not in self.categorized_regions:
                    print(f"Warning: Region {region_name} not loaded, skipping...")
                    continue
                
                # Get region genes
                region_genes = {
                    'normal': self.categorized_regions[region_name]['normal'],
                    'high': self.categorized_regions[region_name]['high']
                }
                
                # Calculate ratio
                ratio = pathway_analyzer.calculate_suppression_addiction_ratio(
                    drug_genes,
                    region_genes,
                    self.addiction_genes,
                    self.suppression_genes
                )
                
                # Store ratio
                all_ratios[drug_name][region_name] = ratio
        
        return all_ratios

    def generate_ratio_heatmap(self, 
                              drug_names: Optional[List[str]] = None,
                              region_names: Optional[List[str]] = None,
                              output_path: Optional[str] = None,
                              drug_type: str = "Analyzed") -> plt.Figure:
        """
        Generate a heatmap of suppression/addiction ratios across drugs and brain regions.
        
        Parameters:
        -----------
        drug_names : List[str], optional
            List of drug names to include. If None, uses all loaded drugs.
        region_names : List[str], optional
            List of brain regions to include. If None, uses all loaded regions.
        output_path : str, optional
            Path to save the heatmap. If None but output_dir is set,
            saves to output_dir/ratio_heatmap.png.
        drug_type : str, default="Analyzed"
            Label for the drug type in the plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with the heatmap
        """
        from .Pathway_analysis import PathwayAnalyzer
        
        # Calculate ratios
        all_ratios = self.calculate_suppression_addiction_ratios(drug_names, region_names)
        
        # Initialize pathway analyzer
        pathway_analyzer = PathwayAnalyzer(output_dir=self.output_dir)
        
        # Set output path if not provided
        if not output_path and self.output_dir:
            output_path = os.path.join(self.output_dir, f"ratio_heatmap_{drug_type.lower().replace(' ', '_')}.png")
        
        # Generate heatmap
        fig = pathway_analyzer.generate_multi_region_heatmap(
            {drug: self.drugs[drug] for drug in all_ratios.keys()},
            list(next(iter(all_ratios.values())).keys()) if all_ratios else [],
            self.addiction_genes,
            self.suppression_genes,
            output_path,
            drug_type
        )
        
        return fig

    def compare_drug_groups(self, 
                           group1_drugs: List[str], 
                           group2_drugs: List[str],
                           group1_label: str = "Group 1",
                           group2_label: str = "Group 2",
                           output_dir: Optional[str] = None) -> Dict:
        """
        Compare two groups of drugs based on their suppression/addiction ratios.
        
        Parameters:
        -----------
        group1_drugs : List[str]
            List of drug names in the first group
        group2_drugs : List[str]
            List of drug names in the second group
        group1_label : str, default="Group 1"
            Label for the first group
        group2_label : str, default="Group 2"
            Label for the second group
        output_dir : str, optional
            Directory to save comparison results
            
        Returns:
        --------
        Dict
            Statistical comparison results
        """
        from .Pathway_analysis import PathwayAnalyzer
        
        # Calculate ratios for both groups
        group1_ratios = self.calculate_suppression_addiction_ratios(group1_drugs)
        group2_ratios = self.calculate_suppression_addiction_ratios(group2_drugs)
        
        # Initialize pathway analyzer
        pathway_analyzer = PathwayAnalyzer(
            output_dir=output_dir if output_dir else self.output_dir
        )
        
        # Compare groups
        results = pathway_analyzer.compare_addictive_nonaddictive_drugs(
            group1_ratios,  # Using addictive parameter for group1
            group2_ratios,  # Using nonaddictive parameter for group2
            output_dir=output_dir if output_dir else self.output_dir
        )
        
        return results

    def save_results(self, results: Dict, filename: str) -> None:
        """
        Save analysis results to a file.
        
        Parameters:
        -----------
        results : Dict
            Analysis results to save
        filename : str
            Name of the file to save to
        """
        import json
        
        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, filename)
        else:
            filepath = filename
        
        # Handle non-serializable types
        def serialize(obj):
            if isinstance(obj, set):
                return list(obj)
            return str(obj)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, default=serialize, indent=2)
        
        print(f"Results saved to {filepath}")

    def process_pathway_reports(self, 
                               addictive_dir: str, 
                               nonaddictive_dir: str) -> Tuple[Dict, Dict]:
        """
        Process pathway report files to extract suppression/addiction ratios.
        
        Parameters:
        -----------
        addictive_dir : str
            Directory containing pathway reports for addictive drugs
        nonaddictive_dir : str
            Directory containing pathway reports for non-addictive drugs
            
        Returns:
        --------
        Tuple[Dict, Dict]
            Dictionaries with ratios for addictive and non-addictive drugs
        """
        from .Pathway_analysis import PathwayAnalyzer
        
        # Initialize pathway analyzer
        pathway_analyzer = PathwayAnalyzer(output_dir=self.output_dir)
        
        # Process reports
        return pathway_analyzer.process_pathway_reports(
            addictive_dir,
            nonaddictive_dir,
            output_subdir="ratio_analysis"
        )
