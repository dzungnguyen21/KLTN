import torch
import torch.nn as nn

# ============================================================
# COMPONENT 2: Hybrid Patch + Region Encoder
# Visual tokens = {patch tokens} ∪ {region tokens}
# ============================================================
class HybridVisualEncoder(nn.Module):
    """
    Kết hợp hai mức trừu tượng:
    - Patch tokens: perception (texture, background, global context)
    - Region tokens: grounding anchor (object-level semantic)

    Trong Qwen2VL, patch tokens đã có sẵn trong visual_hidden_states.
    Region tokens được tổng hợp từ patch tokens qua attention pooling.
    """
    def __init__(self, hidden_size: int, num_regions: int = 16):
        super().__init__()
        self.num_regions = num_regions
        self.hidden_size = hidden_size

        # Learnable region query vectors (như DETR object queries)
        # Mỗi query học cách attend vào một "semantic region" khác nhau
        self.region_queries = nn.Parameter(
            torch.randn(num_regions, hidden_size) * 0.02
        )

        # Cross-attention: region queries attend patch tokens
        self.region_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )

        # Fusion gate: học trọng số blend patch vs region
        self.fusion_gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [batch, num_patches, hidden_size]
        Returns:
            hybrid_tokens: [batch, num_patches + num_regions, hidden_size]
        """
        batch_size = patch_tokens.shape[0]

        # Expand region queries cho batch
        queries = self.region_queries.unsqueeze(0).expand(batch_size, -1, -1)
        # queries: [batch, num_regions, hidden_size]

        # Cross-attention: region queries ← patch tokens
        # Mỗi region query học attend vào spatial region khác nhau
        region_tokens, _ = self.region_attn(
            query=queries,       # [batch, num_regions, H]
            key=patch_tokens,    # [batch, num_patches, H]
            value=patch_tokens   # [batch, num_patches, H]
        )
        # region_tokens: [batch, num_regions, hidden_size]

        # Concatenate: Visual tokens = patches ∪ regions
        hybrid_tokens = torch.cat([patch_tokens, region_tokens], dim=1)
        # [batch, num_patches + num_regions, hidden_size]

        return hybrid_tokens
