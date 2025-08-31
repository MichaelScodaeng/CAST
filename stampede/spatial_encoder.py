# --- START OF FILE stampede/spatial_encoder.py (Fully Vectorized and Corrected) ---

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

from utils.utils import NeighborSampler
# We import the core GNN from the R-PEARL code you provided
from lib.gin import GIN
from lib.mlp import MLP as PEARL_MLP


class DynamicSpatialEncoder(nn.Module):
    """
    Wraps the R-PEARL GNN to make it operate on dynamically sampled
    temporal subgraphs. This version is fully vectorized to avoid slow Python loops.
    """
    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, 
                 device: str, num_neighbors: int, dropout_prob: float = 0.1, bn: bool = True):
        super().__init__()
        self.device = device
        self.num_neighbors = num_neighbors
        self.output_dim = out_dims
        
        def create_pearl_mlp(in_dims, out_dims):
             return PEARL_MLP(n_layers=2, in_dims=in_dims, hidden_dims=in_dims, out_dims=out_dims, 
                              use_bn=False, activation='relu', dropout_prob=dropout_prob)

        self.gnn = GIN(n_layers=n_layers, in_dims=in_dims, hidden_dims=hidden_dims, 
                       out_dims=out_dims, create_mlp=create_pearl_mlp, bn=bn).to(device)
        
        # Project a simple constant to start. The GNN learns to make it meaningful.
        self.node_feature_projection = nn.Linear(1, in_dims)

    def forward(self, node_ids: np.ndarray, node_interact_times: np.ndarray, neighbor_sampler: NeighborSampler):
        """
        For each node in the batch, sample its temporal subgraph, run the GNN,
        and return the root node's embedding in a fully batched manner.
        """
        batch_size = len(node_ids)
        if batch_size == 0:
            return torch.empty((0, self.output_dim), device=self.device)

        # Step 1: Vectorized Neighbor Sampling (CPU)
        # neighbor_nodes shape: [batch_size, num_neighbors]
        neighbor_nodes, _, _ = neighbor_sampler.get_historical_neighbors(
            node_ids, node_interact_times, self.num_neighbors
        )

        # Step 2: Construct a single giant, disconnected graph for the whole batch (PyTorch)
        # The nodes of each subgraph are [root_node, neighbor_1, neighbor_2, ...]
        # We map the global node IDs to a single large tensor
        root_nodes_torch = torch.from_numpy(node_ids).long().unsqueeze(1)
        neighbors_torch = torch.from_numpy(neighbor_nodes).long()
        
        # all_nodes_matrix shape: [batch_size, 1 + num_neighbors]
        all_nodes_matrix = torch.cat([root_nodes_torch, neighbors_torch], dim=1)
        
        # Flatten to get a long list of all nodes involved in the batch
        all_nodes_flat = all_nodes_matrix.flatten()

        # Create initial features for all these nodes
        # We use a simple constant feature; the GNN's job is to create structure-aware embeddings
        x = torch.ones(all_nodes_flat.size(0), 1, device=self.device)
        x = self.node_feature_projection(x)

        # Step 3: Construct the edge_index for the giant graph
        num_subgraph_nodes = 1 + self.num_neighbors
        
        # Create a template for the source and destination of edges within one subgraph
        # The root (local index 0) connects to all neighbors (local indices 1 to num_neighbors)
        edge_index_src_template = torch.zeros(self.num_neighbors, dtype=torch.long)
        edge_index_dst_template = torch.arange(1, self.num_neighbors + 1, dtype=torch.long)
        
        # Create offsets to place each subgraph's edges correctly in the giant graph
        offsets = torch.arange(0, batch_size * num_subgraph_nodes, num_subgraph_nodes, dtype=torch.long)
        
        # Replicate and offset the templates for the entire batch
        batch_edge_index_src = edge_index_src_template.repeat(batch_size) + offsets.repeat_interleave(self.num_neighbors)
        batch_edge_index_dst = edge_index_dst_template.repeat(batch_size) + offsets.repeat_interleave(self.num_neighbors)
        
        # Combine and make the graph undirected
        edge_index = torch.stack([
            torch.cat([batch_edge_index_src, batch_edge_index_dst]),
            torch.cat([batch_edge_index_dst, batch_edge_index_src])
        ], dim=0).to(self.device)

        # Step 4: Run the GNN on the single, large batched graph
        all_node_embeddings = self.gnn(x, edge_index)

        # Step 5: Extract only the embeddings of the original root nodes
        # The root nodes are at indices 0, (1+k), 2*(1+k), ... in the flattened node list
        root_node_indices = torch.arange(0, batch_size * num_subgraph_nodes, num_subgraph_nodes, dtype=torch.long)
        spatial_embeddings = all_node_embeddings[root_node_indices]
        
        return spatial_embeddings
# --- END OF FILE stampede/spatial_encoder.py (Fully Vectorized and Corrected) ---