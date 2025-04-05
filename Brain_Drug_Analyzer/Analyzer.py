import pandas as pd
import numpy as np
import os
from typing import Dict, List, Set, Optional, Union 


class DrugBrainAnalyzer: 

    """
    A class for analyzing the interaction of drug
    with different brain subregions. The current module 
    has been limited to check the interaction with the interaction 
    of drugs with the addiction prone brain regions only,
    but can be extended to include other regions for specific 
    disease of interest.
    """ 

    def __init__(self): 
        self.addiction_genes = set()
        self.suppression_genes = set()
        self.categorized_regions = {} # dictionary for mapping brain regions to categorized genes in the regions
        self.drugs = {} # drugs and their targets which are gathered through the interaction network of STITCH

    def load_gene_sets(self, addiction_gene_file: str, suppression_gene_file: str) -> None: 
        """

        Load the gene sets of addiction and suppression from text files. 

        parameters: 

        ----
        addiction_gene_file: str 
        Path to file that contains addiction-associated genes

        suppression_gene_file: str 
        Path to file that contains suppression-associated genes
        """
        # for addiction genes: 
        with open(addiction_gene_file, 'r') as f: 
            self.addiction_genes = set(line.strip() for line in f)

        # for suppression genes: 

        with open(suppression_gene_file, 'r') as f: 
            self.suppression_genes = set(line.strip() for line in f)
        
        print(f"Loaded {len(self.addiction_genes)} addiction genes and {len(self.suppression_genes)} suppression genes")

    def load_categorized_brain_data(self, brain_directory: str) -> None: 
        
        """
        Load the catgeorized data from the brain directory in the data folder. 

        Parameters :

        ------ 
        brain_directory : str 
        Path to the directory containing the brain regions subdirectories 
        Each brain region is assocaited with addiction potential and should have low, normal and high 
        Each category directory should contain csv file with the gene names
        """

        brain_regions = [d for d in os.listdir(brain_directory) 
                         if os.path.isdir(os.path.join(brain_directory, d))]
        
        for region in brain_regions: 
            region_path = os.path.join(brain_directory, region)

            low_genes = set()
            normal_genes = set()
            high_genes = set()

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

            # clean names for better display as initially the files were done through single scripts: 
            clean_region_name = region.replace("_", " ").replace(",","")

            self.categorized_regions[region] = { 
                'low': low_genes, 
                'normal': normal_genes, 
                'high': high_genes            }
        
        print(f"Loaded the genes based on the expression data for {len(self.categorized_regions)} brain regions")

        


