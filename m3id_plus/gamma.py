import torch
import torch.nn as nn

class LearnableGamma(nn.Module):
    """
    Học adaptive visual trust coefficient từ hidden state.

    γₜ nhỏ → model tự tin vào visual evidence → tăng correction
    γₜ lớn → token ngữ pháp → giảm correction
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # Simple linear projection: W ∈ ℝ^(hidden_size → 1)
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        # Init bias nhỏ để γ bắt đầu ~0.5
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: hidden state [batch, hidden_size]
        Returns:
            gamma: [batch, 1] ∈ (0, 1)
        """
        return torch.sigmoid(self.proj(h_t))  # γₜ = σ(W hₜ)