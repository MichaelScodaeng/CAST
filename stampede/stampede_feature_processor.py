# --- START OF FILE stampede/feature_processor.py (Corrected with Caching) ---

import torch
import torch.nn as nn
import numpy as np

from .fusion_layer import STAMPEDE_FusionLayer
from .temporal_encoder import CombinedLeTE
from models.modules import MergeLayer
from utils.utils import NeighborSampler

class STAMPEDE_Feature_Processor(nn.Module):
    """
    This module is responsible ONLY for feature engineering.
    It takes node IDs and timestamps, and returns a rich initial feature vector.
    It now includes a cache to prevent redundant computations within a single forward pass.
    """
    def __init__(self,
                 spatial_encoder: nn.Module,
                 temporal_encoder: nn.Module,
                 fusion_layer: STAMPEDE_FusionLayer,
                 neighbor_sampler: NeighborSampler,
                 raw_node_feat_dim: int,
                 embedding_type: str = 'all',
                 device: str = 'cpu'):
        super().__init__()
        
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder
        self.fusion_layer = fusion_layer
        self.neighbor_sampler = neighbor_sampler
        self.embedding_type = embedding_type
        self.device = device
        
        # --- Cache for memoization ---
        self.cache = {}
        
        # ... (rest of __init__ is identical) ...
        # Determine the dimension of the combined feature vector
        in_dim = 0
        if self.embedding_type in ['all', 'only_spatial']:
            in_dim += self.fusion_layer.hidden_dim
        if self.embedding_type in ['all', 'only_temporal']:
            in_dim += self.fusion_layer.hidden_dim
        if self.embedding_type in ['all', 'only_st']:
            in_dim += self.fusion_layer.output_dim
        if self.embedding_type == 'none':
            in_dim = raw_node_feat_dim
            
        assert in_dim > 0, "Input dimension for the GNN backbone cannot be zero."
        self.feature_projection = MergeLayer(in_dim, 0, raw_node_feat_dim, raw_node_feat_dim)

    def clear_cache(self):
        """Must be called at the beginning of each new batch forward pass."""
        self.cache = {}

    def forward(self, node_ids: np.ndarray, node_interact_times: np.ndarray, raw_node_features: torch.Tensor):
        
        # Create a unique key for each node-time pair
        # We convert numpy arrays to a tuple of integers for dictionary keys
        keys = tuple(zip(node_ids.tolist(), node_interact_times.tolist()))
        
        # Find which keys are already in the cache and which are new
        cached_indices = [i for i, key in enumerate(keys) if key in self.cache]
        new_indices = [i for i, key in enumerate(keys) if key not in self.cache]
        
        output_features = torch.zeros(len(node_ids), self.feature_projection.output_dim, device=self.device)

        # Retrieve results from cache
        if cached_indices:
            cached_keys = [keys[i] for i in cached_indices]
            cached_values = torch.stack([self.cache[key] for key in cached_keys])
            output_features[torch.tensor(cached_indices)] = cached_values

        # Compute results for new nodes
        if new_indices:
            new_node_ids = node_ids[new_indices]
            new_times = node_interact_times[new_indices]
            new_raw_features = raw_node_features[new_indices]
            
            # 1. Get Spatial Embeddings
            spatial_embeddings = self.spatial_encoder(new_node_ids, new_times, self.neighbor_sampler)
            
            # 2. Get Temporal Embeddings
            timestamps = torch.from_numpy(new_times).float().to(self.device)
            # Add sequence dimension: shape (batch_size,) -> (batch_size, 1)
            timestamps = timestamps.unsqueeze(1)
            temporal_embeddings = self.temporal_encoder(timestamps)
            # Remove sequence dimension: shape (batch_size, 1, dim) -> (batch_size, dim)
            temporal_embeddings = temporal_embeddings.squeeze(1)

            # 3. Get Fused Embeddings
            fused_embeddings = self.fusion_layer(spatial_embeddings, temporal_embeddings)
            
            # 4. Construct feature vector
            if self.embedding_type == 'all':
                initial_features = torch.cat([spatial_embeddings, temporal_embeddings, fused_embeddings], dim=-1)
            elif self.embedding_type == 'only_spatial':
                initial_features = spatial_embeddings
            elif self.embedding_type == 'only_temporal':
                initial_features = temporal_embeddings
            elif self.embedding_type == 'only_st':
                initial_features = fused_embeddings
            elif self.embedding_type == 'none':
                initial_features = new_raw_features
            else:
                raise ValueError(f"Unknown embedding_type: {self.embedding_type}")
            
            # 5. Project to final dimension
            new_features = self.feature_projection(initial_features, torch.empty(0).to(self.device))
            
            # Store new results in cache and in the output tensor
            output_features[torch.tensor(new_indices)] = new_features
            for i, key_idx in enumerate(new_indices):
                self.cache[keys[key_idx]] = new_features[i]

        return output_features

# --- END OF FILE stampede/feature_processor.py (Corrected with Caching) ---