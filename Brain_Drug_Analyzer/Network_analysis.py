import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Optional, Union, Tuple
import io
import time
import json

# API Base URLs
STRING_API_URL = "https://string-db.org/api"
DGIDB_API_URL = "https://dgidb.org/api/v2"


# if the need arises for getting the CID name for the drug: 
def get_pubchem_cid(drug_name):
    """
    Convert a drug name to PubChem CID format used by STRING.
    
    Parameters:
    -----------
    drug_name : str
        Name of the drug
        
    Returns:
    --------
    str
        Formatted CID (CIDmXXXX)
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/JSON"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching PubChem CID for {drug_name}: {response.status_code}")
        return None
    
    try:
        cid = response.json()["IdentifierList"]["CID"][0]
        return f"CIDm{cid}"
    except (KeyError, IndexError) as e:
        print(f"Error parsing PubChem response for {drug_name}: {e}")
        return None


def fetch_dgidb_interactions(drug_name):
    """
    Fetch drug-gene interactions from DGIdb API using GraphQL.
    
    Parameters:
    -----------
    drug_name : str
        Name of the drug
        
    Returns:
    --------
    dict
        JSON response containing interactions
    """
    url = "https://dgidb.org/api/graphql"
    
    # GraphQL query
    query = """
    {
      drugs(names: ["%s"]) {
        nodes {
          name
          interactions {
            gene {
              name
              conceptId
              longName
            }
            interactionScore
            interactionTypes {
              type
              directionality
            }
            publications {
              pmid
            }
            sources {
              sourceDbName
            }
          }
        }
      }
    }
    """ % drug_name
    
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching DGIdb data for {drug_name}: {response.status_code}")
            print(f"Response: {response.text}")
            return {"data": {"drugs": {"nodes": []}}}
        
        return response.json()
        
    except Exception as e:
        print(f"Error in DGIdb API request for {drug_name}: {e}")
        return {"data": {"drugs": {"nodes": []}}}

def fetch_string_interactions(genes, species=9606, confidence=0.7):
    """
    Fetch interactions for a list of genes from STRING.
    
    Parameters:
    -----------
    genes : List[str]
        List of gene symbols
    species : int
        NCBI taxonomy ID (default: 9606 for human)
    confidence : float
        Confidence score threshold (0.0-1.0)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with interaction data
    """
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": int(confidence * 1000),
        "format": "tsv"
    }
    
    response = requests.get(f"{STRING_API_URL}/tsv/network", params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch STRING network data: {response.status_code}")
    
    data = pd.read_csv(io.StringIO(response.text), sep="\t")
    return data

#this was tested but was unsucessful for the time being. 
def fetch_string_chemical_interactions(chemical_id, species=9606, confidence=0.7):
    """
    Fetch interactions for a chemical/drug using STRING API.
    
    Parameters:
    -----------
    chemical_id : str
        Chemical ID in STRING format (CIDmXXXX)
    species : int
        NCBI taxonomy ID (default: 9606 for human)
    confidence : float
        Confidence score threshold (0.0-1.0)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with interaction data
    """
    params = {
        "identifiers": chemical_id,
        "species": species,
        "required_score": int(confidence * 1000),
        "format": "tsv",
        "caller_identity": "network_analysis_tool"
    }
    
    response = requests.get(f"{STRING_API_URL}/tsv/network", params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch STRING chemical network data: {response.status_code}")
    
    data = pd.read_csv(io.StringIO(response.text), sep="\t")
    return data

def build_gene_network(addiction_genes: Set[str], 
                      suppression_genes: Set[str], 
                      confidence_score: float = 0.7, 
                      species: int = 9606) -> nx.Graph:
    """
    Build a gene interaction network with addiction and suppression genes.
    
    Parameters:
    -----------
    addiction_genes : Set[str]
        Set of genes associated with addiction
    suppression_genes : Set[str]
        Set of genes associated with addiction suppression
    confidence_score : float
        Minimum confidence score for interactions (0.0-1.0)
    species : int
        NCBI taxonomy ID (default: 9606 for human)
        
    Returns:
    --------
    nx.Graph
        Gene interaction network
    """
    G = nx.Graph(name="Addiction-Suppression Network")
    
    # Combine gene sets for the API call
    all_genes = list(addiction_genes.union(suppression_genes))
    
    try:
        # Fetch interactions from STRING
        interactions = fetch_string_interactions(all_genes, species, confidence_score)
        
        # Add nodes for all genes
        for gene in addiction_genes:
            G.add_node(gene, type='addiction')
        
        for gene in suppression_genes:
            G.add_node(gene, type='suppression')
        
        # Add interactions as edges
        for _, row in interactions.iterrows():
            gene_a = row['preferredName_A']
            gene_b = row['preferredName_B']
            score = row['score'] / 1000.0  # Convert to 0.0-1.0 scale
            
            # Add edge if both nodes are in our gene sets
            if (gene_a in addiction_genes or gene_a in suppression_genes) and \
               (gene_b in addiction_genes or gene_b in suppression_genes):
                G.add_edge(gene_a, gene_b, weight=score)
    
    except Exception as e:
        print(f"Error building STRING network: {e}")
        # Ensure all genes are added as nodes even if API call fails
        for gene in addiction_genes:
            G.add_node(gene, type='addiction')
        
        for gene in suppression_genes:
            G.add_node(gene, type='suppression')
    
    return G

def build_drug_network(drug_name: str, 
                      confidence_score: float = 0.7) -> nx.Graph:
    """
    Build a drug-gene interaction network using the DGIdb GraphQL API.
    
    Parameters:
    -----------
    drug_name : str
        Name of the drug
    confidence_score : float
        Minimum confidence score for interactions (0.0-1.0)
        Currently unused for DGIdb but kept for API compatibility
        
    Returns:
    --------
    nx.Graph
        Drug-gene interaction network
    """
    G = nx.Graph(name=f"Drug Network: {drug_name}")
    
    # Add drug node
    G.add_node(drug_name, type='drug')
    
    try:
        # Fetch interactions from DGIdb
        response_data = fetch_dgidb_interactions(drug_name)
        
        # Check if we got valid data
        if 'data' not in response_data or 'drugs' not in response_data['data']:
            print(f"Warning: Invalid response format for {drug_name}")
            return G
            
        drug_nodes = response_data['data']['drugs']['nodes']
        if not drug_nodes:
            print(f"Warning: No drug nodes found for {drug_name}")
            return G
            
        # Process each matched drug 
        for drug_node in drug_nodes:
            if 'interactions' not in drug_node:
                continue
                
            # Process each interaction
            for interaction in drug_node['interactions']:
                if 'gene' not in interaction or not interaction['gene']:
                    continue
                    
                gene_info = interaction['gene']
                gene_name = gene_info.get('name')
                
                if not gene_name:
                    continue
                
                # Get interaction types
                interaction_types = []
                for itype in interaction.get('interactionTypes', []):
                    if itype and 'type' in itype:
                        interaction_types.append(itype['type'])
                
                interaction_type = ';'.join(interaction_types) if interaction_types else 'unknown'
                
                # Get sources for confidence assessment
                sources = interaction.get('sources', [])
                source_count = len(sources)
                
                # Use number of sources as a crude confidence measure
                if source_count > 0:
                    # Add gene node
                    G.add_node(gene_name, type='target')
                    
                    # Add edge with metadata
                    G.add_edge(
                        drug_name, 
                        gene_name, 
                        weight=min(1.0, source_count/10.0),  # Normalize to 0-1 range
                        interaction_type=interaction_type,
                        sources=';'.join([s.get('sourceDbName', '') for s in sources if s])
                    )
    
    except Exception as e:
        print(f"Error building DGIdb drug network for {drug_name}: {e}")
    
    return G

def merge_networks(gene_network: nx.Graph, drug_network: nx.Graph) -> nx.Graph:
    """
    Merge the gene network with a drug interaction network.
    
    Parameters:
    -----------
    gene_network : nx.Graph
        Network with addiction and suppression genes
    drug_network : nx.Graph
        Drug interaction network
        
    Returns:
    --------
    nx.Graph
        Merged network
    """
    # Create a new graph for the merged network
    G = nx.Graph(name=f"Merged Network: {gene_network.name} + {drug_network.name}")
    
    # Add all nodes and edges from gene_network
    for node, attrs in gene_network.nodes(data=True):
        G.add_node(node, **attrs)
    
    for u, v, attrs in gene_network.edges(data=True):
        G.add_edge(u, v, **attrs)
    
    # Add all nodes and edges from drug_network
    for node, attrs in drug_network.nodes(data=True):
        if node in G:
            # Node already exists, update attributes
            existing_attrs = G.nodes[node]
            new_attrs = {**existing_attrs, **attrs}
            
            # Handle the 'type' attribute specially
            if 'type' in existing_attrs and 'type' in attrs:
                if existing_attrs['type'] != attrs['type']:
                    new_attrs['type'] = f"{existing_attrs['type']},{attrs['type']}"
            
            nx.set_node_attributes(G, {node: new_attrs})
        else:
            # Add new node
            G.add_node(node, **attrs)
    
    for u, v, attrs in drug_network.edges(data=True):
        if G.has_edge(u, v):
            # Edge already exists, update attributes
            existing_attrs = G.edges[u, v]
            new_attrs = {**existing_attrs, **attrs}
            nx.set_edge_attributes(G, {(u, v): new_attrs})
        else:
            # Add new edge
            G.add_edge(u, v, **attrs)
    
    return G

def extract_drug_subnetwork(network: nx.Graph, drug_name: str, max_distance: int = 2) -> nx.Graph:
    """
    Extract a subnetwork containing the drug and its interactions up to a specified distance.
    
    Parameters:
    -----------
    network : nx.Graph
        The full network to extract from
    drug_name : str
        Name of the drug to extract subnetwork for
    max_distance : int
        Maximum distance from drug to include (default: 2)
        
    Returns:
    --------
    nx.Graph
        Extracted subnetwork
    """
    if drug_name not in network:
        raise ValueError(f"Drug {drug_name} not found in network")
    
    # Get all nodes within max_distance of the drug
    nodes_to_include = set()
    nodes_to_include.add(drug_name)
    
    # Get nodes at each distance level
    for distance in range(1, max_distance + 1):
        current_nodes = set()
        for node in nodes_to_include:
            neighbors = set(network.neighbors(node))
            current_nodes.update(neighbors)
        nodes_to_include.update(current_nodes)
    
    # Create subgraph with selected nodes
    subnetwork = network.subgraph(nodes_to_include).copy()
    subnetwork.name = f"Subnetwork for {drug_name} (distance ≤ {max_distance})"
    
    return subnetwork

def analyze_subnetwork_genes(subnetwork: nx.Graph, 
                           drug_name: str, 
                           addiction_genes: Set[str], 
                           suppression_genes: Set[str]) -> Dict:
    """
    Analyze the genes in the subnetwork and their interactions with the drug.
    
    Parameters:
    -----------
    subnetwork : nx.Graph
        The drug subnetwork to analyze
    drug_name : str
        Name of the drug
    addiction_genes : Set[str]
        Set of genes associated with addiction
    suppression_genes : Set[str]
        Set of genes associated with addiction suppression
        
    Returns:
    --------
    Dict
        Analysis results including counts and lists of interacting genes
    """
    # Get all genes in the subnetwork (excluding the drug)
    all_genes = set(subnetwork.nodes()) - {drug_name}
    
    # Find interacting addiction and suppression genes
    interacting_addiction = all_genes.intersection(addiction_genes)
    interacting_suppression = all_genes.intersection(suppression_genes)
    
    # Calculate statistics
    total_interacting = len(all_genes)
    addiction_count = len(interacting_addiction)
    suppression_count = len(interacting_suppression)
    
    # Calculate ratio (avoid division by zero)
    ratio = suppression_count / addiction_count if addiction_count > 0 else float('inf')
    
    return {
        'total_interacting_genes': total_interacting,
        'addiction_gene_count': addiction_count,
        'suppression_gene_count': suppression_count,
        'suppression_addiction_ratio': ratio,
        'interacting_addiction_genes': sorted(interacting_addiction),
        'interacting_suppression_genes': sorted(interacting_suppression)
    }

def visualize_subnetwork(subnetwork: nx.Graph) -> plt.Figure:
    """
    Create a visualization of the subnetwork.
    
    Parameters:
    -----------
    subnetwork : nx.Graph
        The subnetwork to visualize
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure containing the visualization
    """
    plt.figure(figsize=(12, 10))
    
    # Define node colors based on type
    node_colors = []
    for node in subnetwork.nodes():
        node_type = subnetwork.nodes[node].get('type', 'unknown')
        if 'drug' in str(node_type):
            color = 'yellow'
        elif 'addiction' in str(node_type):
            color = 'red'
        elif 'suppression' in str(node_type):
            color = 'green'
        else:
            color = 'blue'
        node_colors.append(color)
    
    # Draw the network
    pos = nx.spring_layout(subnetwork, k=1, iterations=50)
    nx.draw_networkx_nodes(subnetwork, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(subnetwork, pos, alpha=0.5)
    nx.draw_networkx_labels(subnetwork, pos, font_size=8)
    
    plt.title(subnetwork.name)
    plt.axis('off')
    
    return plt.gcf()