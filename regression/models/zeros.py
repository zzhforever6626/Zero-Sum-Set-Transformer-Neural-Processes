import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroSSetAttention(nn.Module):

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.batch_first = True

        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.s1_scale = nn.Parameter(torch.zeros(nhead))
        self.s1_bias = nn.Parameter(torch.zeros(nhead))
        self.sh_scale = nn.Parameter(torch.zeros(nhead))
        self.sh_bias = nn.Parameter(torch.zeros(nhead))

        hidden_dim = 2 * d_model

        # logit projection

        # self.u_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.u_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=True),
        )

        # Smoothing
        self.mu = nn.Parameter(torch.zeros(nhead, self.d_head))  # [H, Dh]
        self.tau = nn.Parameter(torch.zeros(nhead))  # [H]

        # Query gates
        self.g1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, nhead, bias=True),
        )
        self.gh = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, nhead, bias=True),
        )

        self.dropout = nn.Dropout(dropout)

        self._fallback_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    @staticmethod
    def _is_row_constant_mask(mask: torch.Tensor) -> bool:
        if mask is None:
            return True
        if mask.dim() != 2 or mask.size(0) != mask.size(1):
            return False
        row0 = mask[0:1, :]  # [1, S]
        return torch.equal(mask, row0.expand_as(mask))

    def forward(self,
                src: torch.Tensor,
                src_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None):
        """
        src: [B, S, D]
        src_mask: [S, S]
        """
        B, S, D = src.shape
        device = src.device

        if (src_mask is not None) and (not self._is_row_constant_mask(src_mask)):
            attn_out, _ = self._fallback_mha(src, src, src, attn_mask=src_mask)
            return attn_out

        if src_mask is None:
            key_idx = torch.arange(S, device=device)
        else:
            allowed = (src_mask[0] == 0)  # [S] bool
            key_idx = torch.nonzero(allowed, as_tuple=False).squeeze(-1)
            if key_idx.numel() == 0:
                return torch.zeros_like(src)

        C = key_idx.numel()  # number of keys (context tokens)
        C_float = float(C)

        # Project QKV
        Q = self.q_proj(src)  # [B, S, D]
        K = self.k_proj(src[:, key_idx, :])  # [B, C, D]
        V = self.v_proj(src[:, key_idx, :])  # [B, C, D]

        # Reshape to heads
        Q = Q.view(B, S, self.nhead, self.d_head)  # [B, S, H, Dh]
        K = K.view(B, C, self.nhead, self.d_head)  # [B, C, H, Dh]
        V = V.view(B, C, self.nhead, self.d_head)  # [B, C, H, Dh]

        # Normalize for angular term
        q_norm = Q.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        k_norm = K.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        Q_hat = Q / q_norm  # [B,S,H,Dh]
        K_hat = K / k_norm  # [B,C,H,Dh]

        # # Radial weights
        U_all = self.u_proj(src).view(B, S, self.nhead, self.d_head)  # [B,S,H,Dh]
        U_all_h = U_all.permute(0, 2, 1, 3).contiguous()  # [B,H,S,Dh]

        # U_ctx only for context stats
        U_ctx_h = U_all_h.index_select(dim=2, index=key_idx)  # [B,H,C,Dh]

        # smoothing (context-only)
        exp_tau = torch.exp(self.tau).view(1, self.nhead, 1, 1)  # [1,H,1,1]
        mu = self.mu.view(1, self.nhead, 1, self.d_head)  # [1,H,1,Dh]
        U_sum = U_ctx_h.sum(dim=2, keepdim=True)  # [B,H,1,Dh]
        U_bar = (exp_tau * mu + U_sum) / (exp_tau + C_float)  # [B,H,1,Dh]

        # full s for query gating
        s_all = -(U_all_h * U_bar).sum(dim=-1) / math.sqrt(self.d_head)  # [B,H,S]
        # s_all = (U_all_h * U_all_h).sum(dim=-1, keepdim=True) / self.d_head

        # context s for alpha/delta only
        s_ctx = s_all.index_select(dim=2, index=key_idx)  # [B,H,C]
        alpha = torch.softmax(s_ctx, dim=-1)
        s_mean = s_ctx.mean(dim=-1, keepdim=True)
        delta = s_ctx - s_mean

        # Query gates
        a1 = self.s1_scale.view(1, self.nhead, 1)
        b1 = self.s1_bias.view(1, self.nhead, 1)
        ah = self.sh_scale.view(1, self.nhead, 1)
        bh = self.sh_bias.view(1, self.nhead, 1)
        sigma1_src = self.g1(src).permute(0, 2, 1).contiguous()  # [B,H,S]
        sigmah_src = self.gh(src).permute(0, 2, 1).contiguous()  # [B,H,S]

        sigma1 = torch.sigmoid(sigma1_src + a1 * s_all + b1)
        sigmah = torch.sigmoid(sigmah_src + ah * s_all + bh)

        beta = (sigma1 - sigmah) / C_float  # [B,H,S]
        gamma = (-sigmah) / C_float  # [B,H,S]

        Q_hat_h = Q_hat.permute(0, 2, 1, 3).contiguous()  # [B,H,S,Dh]
        K_hat_h = K_hat.permute(0, 2, 1, 3).contiguous()  # [B,H,C,Dh]
        V_h = V.permute(0, 2, 1, 3).contiguous()  # [B,H,C,Dh]

        # Weighted outer
        A_mat = torch.einsum("bhc,bhcd,bhce->bhde", alpha, K_hat_h, V_h)
        B_mat = torch.einsum("bhc,bhcd,bhce->bhde", delta, K_hat_h, V_h)
        C_mat = torch.einsum("bhcd,bhce->bhde", K_hat_h, V_h)
        sigmah_b = sigmah.unsqueeze(-1).unsqueeze(-1)  # [B,H,S,1,1]
        beta_b = beta.unsqueeze(-1).unsqueeze(-1)
        gamma_b = gamma.unsqueeze(-1).unsqueeze(-1)
        A_exp = A_mat.unsqueeze(2)
        B_exp = B_mat.unsqueeze(2)
        C_exp = C_mat.unsqueeze(2)
        S_mat = sigmah_b * A_exp + beta_b * B_exp + gamma_b * C_exp  # [B,H,S,Dh,Dh]

        # Output
        O_h = torch.einsum("bhsd,bhsde->bhse", Q_hat_h, S_mat)  # [B,H,S,Dh]

        # Reassemble [B,S,D]
        O = O_h.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        O = self.out_proj(O)
        O = self.dropout(O)
        return O


class ZeroSTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu", norm_first=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.norm_first = norm_first

        self.self_attn = ZeroSSetAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, src_mask, src_key_padding_mask):
        x = self.self_attn(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
