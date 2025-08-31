# --- START OF FILE dynamic_spatial_encoder.py ---

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import get_laplacian, to_undirected

from utils.utils import NeighborSampler
# We import the core GNN from the R-PEARL code you provided
from lib.gin import GIN
from lib.mlp import MLP as PEARL_MLP

class DynamicSpatialEncoder(nn.Module):
    """
    Wraps the R-PEARL GNN to make it operate on dynamically sampled
    temporal subgraphs for continuous-time dynamic graphs.
    """
    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, 
                 device: str, dropout_prob: float = 0.1, bn: bool = True):
        super().__init__()
        self.device = device
        
        # This is the core message-passing GNN from R-PEARL
        def create_pearl_mlp(in_dims, out_dims):
             return PEARL_MLP(n_layers=2, in_dims=in_dims, hidden_dims=in_dims, out_dims=out_dims, 
                              use_bn=False, activation='relu', dropout_prob=dropout_prob)

        self.gnn = GIN(n_layers=n_layers, in_dims=in_dims, hidden_dims=hidden_dims, 
                       out_dims=out_dims, create_mlp=create_pearl_mlp, bn=bn).to(device)
        
        self.node_feature_projection = nn.Linear(1, in_dims) # Project a simple constant to start
        self.output_dim = out_dims

    def forward(self, node_ids: np.ndarray, node_interact_times: np.ndarray, neighbor_sampler: NeighborSampler):
        """
        For each node in the batch, sample its temporal subgraph, run the GNN,
        and return the root node's embedding.
        """
        batch_size = len(node_ids)
        spatial_embeddings = []
        
        # Process nodes in smaller chunks to avoid memory issues
        chunk_size = min(32, batch_size)  # Process at most 32 nodes at once
        
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk_node_ids = node_ids[chunk_start:chunk_end]
            chunk_times = node_interact_times[chunk_start:chunk_end]
            
            subgraph_list = []
            root_node_indices = []
            
            for i in range(len(chunk_node_ids)):
                node_id = chunk_node_ids[i]
                ts = chunk_times[i]
                
                # Sample the 1-hop temporal neighborhood
                neighbor_nodes, neighbor_edge_ids, neighbor_times, _ = \
                    neighbor_sampler.find_neighbors_before(node_id, ts)
                
                # Limit number of neighbors to avoid memory explosion
                max_neighbors = 20
                if len(neighbor_nodes) > max_neighbors:
                    neighbor_nodes = neighbor_nodes[:max_neighbors]
                
                # Create a local subgraph
                # The nodes in our subgraph are the target node plus its neighbors
                subgraph_nodes = np.union1d(np.array([node_id]), neighbor_nodes)
                
                # Map global node IDs to local indices (0, 1, 2, ...)
                node_map = {global_id: local_id for local_id, global_id in enumerate(subgraph_nodes)}
                
                if len(neighbor_nodes) > 0:
                    # Create edge_index using local indices
                    source_nodes_local = [node_map[node_id]] * len(neighbor_nodes)
                    neighbor_nodes_local = [node_map[n_id] for n_id in neighbor_nodes]
                    
                    edge_index = torch.tensor([source_nodes_local + neighbor_nodes_local,
                                               neighbor_nodes_local + source_nodes_local], dtype=torch.long).to(self.device)
                else:
                    # Handle isolated nodes
                    edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)

                # Assign a simple initial feature (e.g., a constant 1) to each node in the subgraph
                # The GNN will learn to create meaningful embeddings from this.
                x = torch.ones(len(subgraph_nodes), 1).to(self.device)
                x = self.node_feature_projection(x)

                data = Data(x=x, edge_index=edge_index, num_nodes=len(subgraph_nodes))
                subgraph_list.append(data)
                root_node_indices.append(node_map[node_id])

            # Create a batch of all the small subgraphs for this chunk
            subgraph_batch = Batch.from_data_list(subgraph_list).to(self.device)
            
            # Run the GNN on the entire batch of subgraphs
            with torch.cuda.amp.autocast():  # Use mixed precision to save memory
                all_node_embeddings = self.gnn(subgraph_batch.x, subgraph_batch.edge_index)

            # Extract the embeddings of the original root nodes
            # We need to offset the root_node_indices by the batch pointer
            ptr = subgraph_batch.ptr
            root_offsets = ptr[:-1]
            
            final_root_indices = torch.tensor(root_node_indices, dtype=torch.long).to(self.device) + root_offsets
            
            chunk_spatial_embeddings = all_node_embeddings[final_root_indices]
            spatial_embeddings.append(chunk_spatial_embeddings)
            
            # Clean up intermediate tensors
            del subgraph_batch, all_node_embeddings
            torch.cuda.empty_cache()
        
        # Concatenate all chunk results
        spatial_embeddings = torch.cat(spatial_embeddings, dim=0)
        return spatial_embeddings

# --- END OF FILE dynamic_spatial_encoder.py ---