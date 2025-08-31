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
        input_spatial_dim: int,
        input_temporal_dim: int,
        hidden_dim: int,
        output_dim: int,
        fusion_method: str = 'clifford',
        weighted_fusion_learnable: bool = True,
        input_mlp_layers: int = 2,
        output_mlp_layers: int = 2,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_spatial_dim = input_spatial_dim
        self.input_temporal_dim = input_temporal_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        valid_methods = ['clifford', 'caga', 'weighted', 'concat_mlp', 'cross_attention']
        if fusion_method not in valid_methods:
            raise ValueError(f"fusion_method must be one of {valid_methods}, got {fusion_method}")
        
        # --- Input MLPs: Project to standardized hidden dimension ---
        self.spatial_input_mlp = self._build_mlp(input_spatial_dim, hidden_dim, input_mlp_layers, dropout)
        self.temporal_input_mlp = self._build_mlp(input_temporal_dim, hidden_dim, input_mlp_layers, dropout)

        # --- Fusion-Specific Components ---
        if fusion_method == 'clifford':
            fusion_output_dim = hidden_dim * hidden_dim  # bivector dimension
        
        elif fusion_method == 'caga':
            context_dim = hidden_dim + hidden_dim
            self.mpn = MetricParameterizationNetwork(context_dim, hidden_dim, hidden_dim, dropout)
            fusion_output_dim = 1 + (hidden_dim * hidden_dim)  # scalar + bivector
            
        elif fusion_method == 'weighted':
            if weighted_fusion_learnable:
                self.spatial_weight = nn.Parameter(torch.tensor(0.5))
            else:
                self.register_buffer('spatial_weight', torch.tensor(0.5))
            fusion_output_dim = hidden_dim
            
        elif fusion_method == 'concat_mlp':
            """concat_dim = self.spatial_dim + self.temporal_dim
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
            self.mlp = nn.Sequential(*layers)"""
            fusion_output_dim = hidden_dim + hidden_dim

        elif fusion_method == 'cross_attention':
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
                batch_first=True
            )
            fusion_output_dim = hidden_dim

        # --- Output MLP: Project from fusion output to final dimension ---
        self.output_mlp = self._build_mlp(fusion_output_dim, output_dim, output_mlp_layers, dropout)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)

        self._init_weights()
    
    def _build_mlp(self, input_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Module:
        """Build a multi-layer perceptron with specified architecture."""
        if num_layers == 1:
            return nn.Linear(input_dim, output_dim)
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            if i == 0:
                hidden_dim = max(input_dim, output_dim)
            else:
                hidden_dim = max(current_dim // 2, output_dim)
            
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, spatial_embedding: torch.Tensor, temporal_embedding: torch.Tensor) -> torch.Tensor:
        # Step 1: Project inputs to standardized hidden dimension
        spatial_vec = self.spatial_input_mlp(spatial_embedding)
        temporal_vec = self.temporal_input_mlp(temporal_embedding)
        
        # Step 2: Apply fusion strategy
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
        
        # Step 3: Project to final output dimension
        output = self.output_mlp(fused)
        return self.layer_norm(output)

    def _clifford_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        # Efficiently compute the outer product for a batch.
        # spatial_vec: [batch, hidden_dim], temporal_vec: [batch, hidden_dim]
        # Reshape for batch matrix multiplication: [batch, hidden_dim, 1] @ [batch, 1, hidden_dim] -> [batch, hidden_dim, hidden_dim]
        interaction_matrix = torch.bmm(spatial_vec.unsqueeze(2), temporal_vec.unsqueeze(1))
        # Flatten the resulting bivector coefficients
        bivector_coeffs = interaction_matrix.view(spatial_vec.size(0), -1)
        return bivector_coeffs

    def _caga_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        # 1. Learn the event-specific metric (inner product) from context
        context = torch.cat([spatial_vec, temporal_vec], dim=-1)
        st_inner_products = self.mpn(context) # [batch, hidden_dim * hidden_dim]

        # 2. Compute the scalar part (inner product) of the geometric product
        # Reshape inner products to matrix form for element-wise multiplication
        st_inner_matrix = st_inner_products.view(-1, self.hidden_dim, self.hidden_dim)
        # Element-wise product and sum to get the final scalar value
        scalar_part = torch.einsum('bi,bij,bj->b', spatial_vec, st_inner_matrix, temporal_vec).unsqueeze(-1) # [batch, 1]

        # 3. Compute the bivector part (outer product) of the geometric product
        interaction_matrix = torch.bmm(spatial_vec.unsqueeze(2), temporal_vec.unsqueeze(1))
        bivector_part = interaction_matrix.view(spatial_vec.size(0), -1) # [batch, hidden_dim * hidden_dim]

        # 4. Concatenate all grade parts
        full_multivector = torch.cat([scalar_part, bivector_part], dim=-1)
        return full_multivector

    def _weighted_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        if isinstance(self.spatial_weight, nn.Parameter):
            # Normalize learnable weights to sum to 1
            temporal_weight = 1.0 - self.spatial_weight
            weighted_embedding = self.spatial_weight * spatial_vec + temporal_weight * temporal_vec
        else: # Fixed weights
            weighted_embedding = self.spatial_weight * spatial_vec + (1.0 - self.spatial_weight) * temporal_vec
        
        return weighted_embedding

    def _concat_mlp_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        """Perform concatenation + MLP fusion."""
        # This method's only job is to concatenate. The output_mlp will do the processing.
        return torch.cat([spatial_vec, temporal_vec], dim=-1)
    def _cross_attention_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention expects input as (Batch, SeqLen, Dim)
        spatial_seq = spatial_vec.unsqueeze(1)
        temporal_seq = temporal_vec.unsqueeze(1)
        
        # Spatial attends to temporal context
        attended_spatial, _ = self.cross_attention(query=spatial_seq, key=temporal_seq, value=temporal_seq)
        
        return attended_spatial.squeeze(1)

# --- END OF FILE fusion_strat.py ---