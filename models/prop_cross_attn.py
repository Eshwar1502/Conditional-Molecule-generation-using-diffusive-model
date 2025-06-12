import torch
import torch.nn as nn
import torch.nn.functional as F

class PropertyCrossAttention(nn.Module):
    """Dedicated cross-attention between node features and properties"""
    def __init__(self, node_dim, prop_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        # Property projections
        self.k_proj = nn.Linear(prop_dim, node_dim)
        self.v_proj = nn.Linear(prop_dim, node_dim)
        
        # Node projections
        self.q_proj = nn.Linear(node_dim, node_dim)
        
        # Output
        self.out_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.LayerNorm(node_dim))
        
    def forward(self, node_feats, properties):
        """
        Args:
            node_feats: [B, N, D] 
            properties: [B, D_prop]
        """
        B, N, D = node_feats.shape
        
        # Project properties
        k = self.k_proj(properties).unsqueeze(1)  # [B, 1, D]
        v = self.v_proj(properties).unsqueeze(1)  # [B, 1, D]
        
        # Project queries
        q = self.q_proj(node_feats)  # [B, N, D]
        
        # Split heads
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = F.softmax(
            (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim)),
            dim=-1
        )
        
        # Combine
        context = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(context + node_feats)