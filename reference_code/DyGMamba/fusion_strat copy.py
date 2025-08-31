# --- START OF FILE fusion_strat.py ---

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class MetricParameterizationNetwork(nn.Module):
    """A simple MLP to learn the metric for CAGA fusion."""
    def __init__(self, input_dim: int, spatial_dim: int, temporal_dim: int, dropout: float):
        super().__init__()
        hidden_dim = (input_dim + (spatial_dim * temporal_dim)) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, spatial_dim * temporal_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns the inner product coefficients for the ST part of the metric
        return self.net(x)


class STAMPEDE_FusionLayer(nn.Module):
    """
    Core Spatiotemporal Fusion Layer for the STAMPEDE Framework.
    
    Supports five fusion methods:
    1. 'clifford' (C-CASF): Clifford algebra-based fusion with a fixed metric.
    2. 'caga': Context-Adaptive Geometric Algebra fusion with a learned metric.
    3. 'weighted': Weighted summation of spatial and temporal embeddings.
    4. 'concat_mlp': Concatenation followed by MLP projection.
    5. 'cross_attention': Cross-attention between spatial and temporal embeddings.
    """
    
    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int, 
        output_dim: int,
        input_spatial_dim: Optional[int] = None,
        input_temporal_dim: Optional[int] = None,
        fusion_method: str = 'clifford',
        weighted_fusion_learnable: bool = True,
        mlp_hidden_dim: Optional[int] = None,
        mlp_num_layers: int = 2,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        valid_methods = ['clifford', 'caga', 'weighted', 'concat_mlp', 'cross_attention']
        if fusion_method not in valid_methods:
            raise ValueError(f"fusion_method must be one of {valid_methods}, got {fusion_method}")
        
        # --- Input Projections ---
        # Project spatial embedding to target dimension if necessary
        self.spatial_projection = nn.Linear(input_spatial_dim, spatial_dim) if input_spatial_dim and input_spatial_dim != spatial_dim else nn.Identity()
        # Project temporal embedding to target dimension if necessary
        self.temporal_projection = nn.Linear(input_temporal_dim, temporal_dim) if input_temporal_dim and input_temporal_dim != temporal_dim else nn.Identity()

        # --- Method-Specific Setups ---
        if fusion_method == 'clifford':
            bivector_dim = self.spatial_dim * self.temporal_dim
            self.output_projection = nn.Linear(bivector_dim, self.output_dim)
        
        elif fusion_method == 'caga':
            context_dim = self.spatial_dim + self.temporal_dim
            self.mpn = MetricParameterizationNetwork(context_dim, self.spatial_dim, self.temporal_dim, dropout)
            # Output will be scalar (inner prod) + bivector (outer prod)
            caga_input_dim = 1 + (self.spatial_dim * self.temporal_dim)
            self.output_projection = nn.Linear(caga_input_dim, self.output_dim)

        elif fusion_method == 'weighted':
            assert self.spatial_dim == self.temporal_dim, "For weighted fusion, spatial_dim and temporal_dim must be equal."
            if weighted_fusion_learnable:
                self.spatial_weight = nn.Parameter(torch.tensor(0.5))
            else:
                self.register_buffer('spatial_weight', torch.tensor(0.5))
            self.output_projection = nn.Linear(self.spatial_dim, self.output_dim)

        elif fusion_method == 'concat_mlp':
            concat_dim = self.spatial_dim + self.temporal_dim
            hidden_dim = hidden_dim if mlp_hidden_dim is not None else concat_dim * 2
            layers = [
                nn.Linear(concat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            for _ in range(mlp_num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            layers.append(nn.Linear(hidden_dim, self.output_dim))
            self.mlp = nn.Sequential(*layers)

        elif fusion_method == 'cross_attention':
            assert self.spatial_dim == self.temporal_dim, "For cross-attention, spatial_dim and temporal_dim must be equal."
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.spatial_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
                batch_first=True
            )
            self.output_projection = nn.Linear(self.spatial_dim, self.output_dim)
            
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)

        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, spatial_embedding: torch.Tensor, temporal_embedding: torch.Tensor) -> torch.Tensor:
        spatial_vec = self.spatial_projection(spatial_embedding)
        temporal_vec = self.temporal_projection(temporal_embedding)
        
        if self.fusion_method == 'clifford':
            fused = self._clifford_fusion(spatial_vec, temporal_vec)
        elif self.fusion_method == 'caga':
            fused = self._caga_fusion(spatial_vec, temporal_vec)
        elif self.fusion_method == 'weighted':
            fused = self._weighted_fusion(spatial_vec, temporal_vec)
        elif self.fusion_method == 'concat_mlp':
            fused = self._concat_mlp_fusion(spatial_vec, temporal_vec)
        elif self.fusion_method == 'cross_attention':
            fused = self._cross_attention_fusion(spatial_vec, temporal_vec)
        
        return self.layer_norm(fused)

    def _clifford_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        # Efficiently compute the outer product for a batch.
        # spatial_vec: [batch, D_s], temporal_vec: [batch, D_t]
        # Reshape for batch matrix multiplication: [batch, D_s, 1] @ [batch, 1, D_t] -> [batch, D_s, D_t]
        interaction_matrix = torch.bmm(spatial_vec.unsqueeze(2), temporal_vec.unsqueeze(1))
        # Flatten the resulting bivector coefficients
        bivector_coeffs = interaction_matrix.view(spatial_vec.size(0), -1)
        return self.output_projection(bivector_coeffs)

    def _caga_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        # 1. Learn the event-specific metric (inner product) from context
        context = torch.cat([spatial_vec, temporal_vec], dim=-1)
        st_inner_products = self.mpn(context) # [batch, D_s * D_t]

        # 2. Compute the scalar part (inner product) of the geometric product
        # Reshape inner products to matrix form for element-wise multiplication
        st_inner_matrix = st_inner_products.view(-1, self.spatial_dim, self.temporal_dim)
        # Element-wise product and sum to get the final scalar value
        scalar_part = torch.einsum('bi,bij,bj->b', spatial_vec, st_inner_matrix, temporal_vec).unsqueeze(-1) # [batch, 1]

        # 3. Compute the bivector part (outer product) of the geometric product
        interaction_matrix = torch.bmm(spatial_vec.unsqueeze(2), temporal_vec.unsqueeze(1))
        bivector_part = interaction_matrix.view(spatial_vec.size(0), -1) # [batch, D_s * D_t]

        # 4. Concatenate all grade parts and project
        full_multivector = torch.cat([scalar_part, bivector_part], dim=-1)
        return self.output_projection(full_multivector)

    def _weighted_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        if isinstance(self.spatial_weight, nn.Parameter):
            # Normalize learnable weights to sum to 1
            temporal_weight = 1.0 - self.spatial_weight
            weighted_embedding = self.spatial_weight * spatial_vec + temporal_weight * temporal_vec
        else: # Fixed weights
            weighted_embedding = self.spatial_weight * spatial_vec + (1.0 - self.spatial_weight) * temporal_vec
        
        return self.output_projection(weighted_embedding)

    def _concat_mlp_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        concat_embedding = torch.cat([spatial_vec, temporal_vec], dim=-1)
        return self.mlp(concat_embedding)

    def _cross_attention_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention expects input as (Batch, SeqLen, Dim)
        spatial_seq = spatial_vec.unsqueeze(1)
        temporal_seq = temporal_vec.unsqueeze(1)
        
        # Spatial attends to temporal context
        attended_spatial, _ = self.cross_attention(query=spatial_seq, key=temporal_seq, value=temporal_seq)
        
        return self.output_projection(attended_spatial.squeeze(1))

# --- END OF FILE fusion_strat.py ---