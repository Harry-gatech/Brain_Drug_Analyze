import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Network_analysis import (
    build_gene_network,
    build_drug_network,
    merge_networks,
    extract_drug_subnetwork,
    analyze_subnetwork_genes,
    visualize_subnetwork
)
import matplotlib.pyplot as plt
import networkx as nx

def run_network_analysis_example():
    """
    Example function demonstrating how to use the network analysis module
    to create and analyze drug interaction networks.
    """
    # Sample data
    drug_name = "fentanyl"
    addiction_genes = {
        "OPRM1", "SLC6A3", "SLC6A4", "SIGMAR1", "CXCR4", 
        "DRD2", "OPRK1", "OPRD1", "CNR1", "GABBR2"
    }
    suppression_genes = {
        "KCNJ6", "DRD2", "GABRA2", "GRIN1", "CHRM5",
        "POMC", "IL1B", "TNF", "NFKB1", "ESR1",
        "FAAH", "TRPV1", "BCL2", "MAPK3", "AKR1B1"
    }
    
    # Step 1: Build the base gene network with addiction and suppression genes
    print(f"Building base network with {len(addiction_genes)} addiction genes and {len(suppression_genes)} suppression genes...")
    gene_network = build_gene_network(addiction_genes, suppression_genes, confidence_score=0.4)
    print(f"Base network created with {len(gene_network.nodes())} nodes and {len(gene_network.edges())} edges")
    
    # Visualize the base network
    plt.figure(figsize=(10, 8))
    node_colors = ['red' if gene in addiction_genes else 'green' for gene in gene_network.nodes()]
    nx.draw_networkx(gene_network, node_color=node_colors, with_labels=True, node_size=500, font_size=8)
    plt.title("Base Network: Addiction and Suppression Genes")
    plt.axis('off')
    plt.savefig("base_network.png")
    plt.close()
    
    # Step 2: Build the drug network using DGIdb API
    print(f"Building drug network for {drug_name}...")
    drug_network = build_drug_network(drug_name, confidence_score=0.4)
    print(f"Drug network created with {len(drug_network.nodes())} nodes and {len(drug_network.edges())} edges")
    
    # Visualize the drug network
    plt.figure(figsize=(10, 8))
    drug_node_colors = ['yellow' if node == drug_name else 'blue' for node in drug_network.nodes()]
    nx.draw_networkx(drug_network, node_color=drug_node_colors, with_labels=True, node_size=500, font_size=8)
    plt.title(f"{drug_name} Drug Network")
    plt.axis('off')
    plt.savefig("drug_network.png")
    plt.close()
    
    # Step 3: Merge the base network and drug network
    print("Merging networks...")
    merged_network = merge_networks(gene_network, drug_network)
    print(f"Merged network created with {len(merged_network.nodes())} nodes and {len(merged_network.edges())} edges")
    
    # Step 4: Extract the subnetwork with first and second interactions
    print("Extracting subnetwork with first and second interactions...")
    
    # First, extract just the direct (first) interactions
    first_level_subnetwork = extract_drug_subnetwork(merged_network, drug_name, max_distance=1)
    print(f"First-level subnetwork: {len(first_level_subnetwork.nodes())} nodes, {len(first_level_subnetwork.edges())} edges")
    
    # Then, extract both first and second interactions
    subnetwork = extract_drug_subnetwork(merged_network, drug_name, max_distance=2)
    print(f"Complete subnetwork (1st & 2nd interactions): {len(subnetwork.nodes())} nodes, {len(subnetwork.edges())} edges")
    
    # Step 5: Analyze which addiction and suppression genes interact with the drug
    print("Analyzing interaction patterns...")
    analysis = analyze_subnetwork_genes(subnetwork, drug_name, addiction_genes, suppression_genes)
    
    # Print analysis results
    print("\nAnalysis Results:")
    print(f"Drug: {drug_name}")
    print(f"Total interacting genes: {analysis['total_interacting_genes']}")
    print(f"Addiction genes: {analysis['addiction_gene_count']}")
    print(f"Suppression genes: {analysis['suppression_gene_count']}")
    print(f"Suppression/Addiction Ratio: {analysis['suppression_addiction_ratio']}")
    
    print("\nInteracting Addiction Genes:")
    for gene in analysis['interacting_addiction_genes']:
        print(f"- {gene}")
    
    print("\nInteracting Suppression Genes:")
    for gene in analysis['interacting_suppression_genes']:
        print(f"- {gene}")
    
    # Step 6: Visualize the final subnetwork
    print("\nCreating visualization...")
    fig = visualize_subnetwork(subnetwork)
    fig.savefig("drug_subnetwork.png")
    
    print("\nAnalysis complete! Network visualizations saved to:")
    print("- base_network.png")
    print("- drug_network.png")
    print("- drug_subnetwork.png")
    
    return analysis, subnetwork

if __name__ == "__main__":
    try:
        run_network_analysis_example()
    except Exception as e:
        print(f"Error running example: {e}")