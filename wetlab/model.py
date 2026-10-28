from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionClassifierGated(nn.Module):
    """MuNaP fusion and classification head used during adaptation."""

    def __init__(
        self,
        x_dim: int,
        pro_seq_dim: int,
        pro_str_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.x_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_seq = nn.Sequential(
            nn.Linear(pro_seq_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_str = nn.Sequential(
            nn.Linear(pro_str_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pro_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.pro_mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout / 2,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 16, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.x_mlp, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, nonlinearity="relu"
                    )
                    nn.init.constant_(layer.bias, 0.1)
        for module in [
            self.proj_seq,
            self.proj_str,
            *[layer for layer in self.pro_gate if isinstance(layer, nn.Linear)],
        ]:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0.0)
        last_gate_layer = [
            layer for layer in self.pro_gate if isinstance(layer, nn.Linear)
        ][-1]
        nn.init.constant_(last_gate_layer.bias, -1.0)

    def forward(self, x_embed, pro_seq_embed, pro_str_embed):
        x_feat = self.x_mlp(x_embed)
        seq_h = self.proj_seq(pro_seq_embed)
        str_h = self.proj_str(pro_str_embed)
        gate = self.pro_gate(torch.cat([seq_h, str_h], dim=-1))
        pro_fused = seq_h + gate * (str_h - seq_h)
        pro_feat = self.pro_mlp(pro_fused)

        x_token = x_feat.unsqueeze(1)
        pro_token = pro_feat.unsqueeze(1)
        attn_out, _ = self.attn(
            query=x_token,
            key=pro_token,
            value=pro_token,
            need_weights=False,
        )
        fused = (x_token + attn_out).squeeze(1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, gate


def finetune_head_only(model: nn.Module) -> None:
    """Freeze the fusion network and enable gradients only on the classifier."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model.eval()
    model.classifier.train()
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"[Finetune head-only] Trainable params: {trainable}/{total}")
