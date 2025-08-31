# --- START OF FILE models/TGAT.py (Modified) ---

import numpy as np
import torch
import torch.nn as nn
from typing import Optional # <<< ADD THIS IMPORT

from models.modules import TimeEncoder, MergeLayer, MultiHeadAttention
from utils.utils import NeighborSampler


class TGAT(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu',
                 feature_preprocessor: Optional[nn.Module] = None): # <<< ADD NEW ARGUMENT
        super(TGAT, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        
        self.feature_preprocessor = feature_preprocessor # <<< STORE THE PREPROCESSOR

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout) for _ in range(num_layers)])
        self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                                                      hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        # This method remains unchanged
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    current_layer_num=self.num_layers, num_neighbors=num_neighbors)
        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         current_layer_num: int, num_neighbors: int = 20):
        assert current_layer_num >= 0
        device = self.node_raw_features.device

        node_time_features = self.time_encoder(timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        
        # --- THIS IS THE KEY MODIFICATION ---
        # Get the raw features for the current nodes
        raw_features = self.node_raw_features[torch.from_numpy(node_ids).long()]
        # If a feature preprocessor (our STAMPEDE module) is provided, use it.
        # Otherwise, fall back to the original raw features.
        if self.feature_preprocessor is not None:
            node_features = self.feature_preprocessor(node_ids, node_interact_times, raw_features)
        else:
            node_features = raw_features
        # --- END OF MODIFICATION ---

        if current_layer_num == 0:
            return node_features
        else:
            # ... (the rest of the function remains identical, it will now propagate our rich features) ...
            node_conv_features = self.compute_node_temporal_embeddings(node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)

            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)
            
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(),
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)
            
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(node_ids.shape[0], num_neighbors, self.node_feat_dim)

            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times
            neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids)

            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)

            return output

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        # This method remains unchanged
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

# --- END OF FILE models/TGAT.py (Modified) ---