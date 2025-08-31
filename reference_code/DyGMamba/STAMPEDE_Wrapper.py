# --- START OF FILE STAMPEDE_Wrapper.py ---

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

# Assuming the following modules are in the path
from models.modules import TimeEncoder, MultiHeadAttention, MergeLayer
from utils.utils import NeighborSampler
# Import our new modules
from stampede.fusion_layer import STAMPEDE_FusionLayer
from LeTE import CombinedLeTE


class STAMPEDE_Wrapper(nn.Module):
    """
    STAMPEDE Wrapper that acts as a complete dynamic backbone.
    It orchestrates spatial encoding, temporal encoding, and fusion before
    performing its own GNN-based message passing.
    """
    def __init__(self,
                 # Sub-modules
                 spatial_encoder: nn.Module,
                 temporal_encoder: nn.Module,
                 fusion_layer: STAMPEDE_FusionLayer,
                 # GNN backbone parameters
                 node_raw_features: torch.Tensor,
                 edge_raw_features: torch.Tensor,
                 neighbor_sampler: NeighborSampler,
                 time_feat_dim: int,
                 num_layers: int,
                 num_heads: int,
                 dropout: float,
                 # STAMPEDE specific parameters
                 embedding_type: str = 'all',
                 device: str = 'cpu'):
        super().__init__()

        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder
        self.fusion_layer = fusion_layer
        self.neighbor_sampler = neighbor_sampler
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.embedding_type = embedding_type
        self.device = device
        
        # --- Determine the dimension of the feature vector before GNN layers ---
        in_dim = 0
        if self.embedding_type in ['all', 'only_spatial']:
            in_dim += self.fusion_layer.spatial_dim
        if self.embedding_type in ['all', 'only_temporal']:
            in_dim += self.fusion_layer.temporal_dim
        if self.embedding_type in ['all', 'only_st']:
            in_dim += self.fusion_layer.output_dim
        if self.embedding_type == 'none':
            # If no embeddings are used, the GNN will operate on raw features
            in_dim = self.node_feat_dim
            
        assert in_dim > 0, "Input dimension for the GNN backbone cannot be zero."

        # Projection layer to map the combined feature vector to the GNN's hidden dimension
        self.feature_projection = MergeLayer(in_dim, 0, self.node_feat_dim, self.node_feat_dim)

        # --- GNN Backbone Layers (similar to TGAT) ---
        self.temporal_conv_layers = nn.ModuleList([
            MultiHeadAttention(
                node_feat_dim=self.node_feat_dim,
                edge_feat_dim=self.edge_feat_dim,
                time_feat_dim=self.time_feat_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.merge_layers = nn.ModuleList([
            MergeLayer(
                input_dim1=self.node_feat_dim + self.time_feat_dim, 
                input_dim2=self.node_feat_dim,
                hidden_dim=self.node_feat_dim, 
                output_dim=self.node_feat_dim
            ) for _ in range(num_layers)
        ])

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        Main forward pass for computing final embeddings for source and destination nodes.
        This method is the standard interface called by the training script.
        """
        # Compute embeddings for both source and destination nodes
        src_node_embeddings = self.compute_node_temporal_embeddings(
            node_ids=src_node_ids, 
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers, 
            num_neighbors=num_neighbors
        )
        dst_node_embeddings = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids, 
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers, 
            num_neighbors=num_neighbors
        )
        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        Recursively computes temporal embeddings for a set of nodes.
        """
        assert current_layer_num >= 0
        
        # Base case:
        if current_layer_num == 0:
            # Generate initial features based on the embedding_type
            
            # 1. Get Spatial Embeddings (e_s)
            # We assume the spatial encoder is a lookup table of pre-computed embeddings
            spatial_embeddings = self.spatial_encoder(torch.from_numpy(node_ids).long().to(self.device))
            
            # 2. Get Temporal Embeddings (e_t)
            timestamps = torch.from_numpy(node_interact_times).float().to(self.device)
            temporal_embeddings = self.temporal_encoder(timestamps)

            # 3. Get Fused Spatio-Temporal Embeddings (e_st)
            fused_embeddings = self.fusion_layer(spatial_embeddings, temporal_embeddings)

            # 4. Construct final feature vector based on configuration
            if self.embedding_type == 'all':
                initial_features = torch.cat([spatial_embeddings, temporal_embeddings, fused_embeddings], dim=-1)
            elif self.embedding_type == 'only_spatial':
                initial_features = spatial_embeddings
            elif self.embedding_type == 'only_temporal':
                initial_features = temporal_embeddings
            elif self.embedding_type == 'only_st':
                initial_features = fused_embeddings
            elif self.embedding_type == 'none':
                initial_features = self.node_raw_features[torch.from_numpy(node_ids).long()]
            else:
                raise ValueError(f"Unknown embedding_type: {self.embedding_type}")
                
            # 5. Project features to the model's hidden dimension
            return self.feature_projection(initial_features, torch.empty(0))

        # Recursive step:
        else:
            # Get embeddings from the previous layer
            node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                current_layer_num=current_layer_num - 1,
                num_neighbors=num_neighbors
            )
            
            # Sample neighbors
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(
                    node_ids=node_ids,
                    node_interact_times=node_interact_times,
                    num_neighbors=num_neighbors
                )

            # Get neighbor embeddings from the previous layer
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=neighbor_node_ids.flatten(),
                node_interact_times=neighbor_times.flatten(),
                current_layer_num=current_layer_num - 1,
                num_neighbors=num_neighbors
            ).view(len(node_ids), num_neighbors, -1)

            # Get time and edge features for the neighborhood
            node_time_features = self.temporal_encoder.time_encoder(torch.zeros(len(node_ids), 1).to(self.device))
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times
            neighbor_time_features = self.temporal_encoder.time_encoder(torch.from_numpy(neighbor_delta_times).float().to(self.device))
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids).long()]

            # Perform attention-based message passing (TGAT layer)
            attention_output, _ = self.temporal_conv_layers[current_layer_num - 1](
                node_features=node_conv_features,
                node_time_features=node_time_features,
                neighbor_node_features=neighbor_node_conv_features,
                neighbor_node_time_features=neighbor_time_features,
                neighbor_node_edge_features=neighbor_edge_features,
                neighbor_masks=neighbor_node_ids
            )
            
            # Merge with initial node features (residual connection)
            final_features = self.merge_layers[current_layer_num - 1](
                input_1=attention_output,
                input_2=node_conv_features
            )

            return final_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        Updates the neighbor sampler.
        """
        self.neighbor_sampler = neighbor_sampler

# --- END OF FILE STAMPEDE_Wrapper.py ---