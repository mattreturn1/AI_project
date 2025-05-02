import numpy as np
import networkx as nx
from scipy.io import loadmat

# Function to load and process correlation matrix
def load_and_process_matrix(file_path):
    """
    Load the correlation matrix, apply linear transformation, and normalize weights.
    """
    data = loadmat(file_path)
    matrix = None
    for key in data.keys():
        if isinstance(data[key], np.ndarray) and data[key].shape[0] == data[key].shape[1]:
            matrix = data[key]
            break
    if matrix is None:
        raise ValueError("Correlation matrix not found in the file.")

    # Apply linear transformation: map [-1, 1] to [0, 1]
    matrix_transformed = (matrix + 1) / 2.0
    np.fill_diagonal(matrix_transformed, 0)  # Remove self-loops

    # Normalize edge weights
    max_weight = np.max(matrix_transformed)
    if max_weight > 0:
        matrix_transformed = matrix_transformed / max_weight

    return matrix_transformed

# Function to create graph from a weighted adjacency matrix
def create_weighted_graph(matrix):
    """
    Create a NetworkX graph from the weighted adjacency matrix.
    """
    graph = nx.from_numpy_array(matrix)
    return graph

# Function to compute clustering coefficients
def compute_clustering_coefficients(graph):
    """
    Compute the weighted clustering coefficients for all nodes.
    """
    clustering = nx.clustering(graph, weight='weight')
    return clustering

# Function to compute closeness centrality
def compute_closeness_centrality(graph):
    """
    Compute the closeness centrality for all nodes.
    """
    closeness = nx.closeness_centrality(graph, distance='weight')
    return closeness

# Function to compute degree centrality
def compute_degree_centrality(graph):
    """
    Compute the degree centrality for all nodes.
    """
    degree_centrality = {node: sum(weight for _, _, weight in graph.edges(node, data='weight')) for node in graph.nodes()}
    return degree_centrality
