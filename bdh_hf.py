import dataclasses
from typing import Optional, Tuple

import torch
from torch import nn

import bdh as bdh_core


class BytesTokenizer:
    """Minimal byte-level tokenizer compatible with BDH (vocab_size=256)."""

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(bytearray(text, "utf-8"), dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return bytes(ids.to(torch.uint8).tolist()).decode(errors="backslashreplace")


@dataclasses.dataclass
class BDHHFConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


class BDHRecurrent(nn.Module):
    """
    Hugging Face-style wrapper exposing a recurrent/stateful interface over BDH.

    Maintains two states per step:
    - global_state: aggregated narrative embedding
    - char_state: target character embedding (updated only if present)
    """

    def __init__(self, config: Optional[BDHHFConfig] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = config or BDHHFConfig()
        core_cfg = bdh_core.BDHConfig(
            n_layer=cfg.n_layer,
            n_embd=cfg.n_embd,
            dropout=0.0,  # No dropout during inference
            n_head=cfg.n_head,
            mlp_internal_dim_multiplier=cfg.mlp_internal_dim_multiplier,
            vocab_size=cfg.vocab_size,
        )
        self.core = bdh_core.BDH(core_cfg).to(self.device)
        self.core.eval()  # Set to eval mode for inference
        self.tokenizer = BytesTokenizer()

        D = cfg.n_embd
        # Simple gated recurrent update for states
        self.global_update = nn.GRUCell(D, D)
        self.char_update = nn.GRUCell(D, D)
        # Projection to a compact embedding for cosine similarity
        self.proj = nn.Linear(D, D)

    def init_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        D = self.core.config.n_embd
        zero_g = torch.zeros(D, device=self.device)
        zero_c = torch.zeros(D, device=self.device)
        return zero_g, zero_c

    @torch.no_grad()
    def step(
        self,
        text: str,
        prev_global: torch.Tensor,
        prev_char: torch.Tensor,
        character_present: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ingest a chunk, update global and character states, return projected embedding.
        """
        ids = self.tokenizer.encode(text).unsqueeze(0).to(self.device)  # (1, T)
        # Truncate to max 512 tokens for speed
        if ids.shape[1] > 512:
            ids = ids[:, :512]
        hidden = self.core.encode(ids)  # (1, T, D)
        # Use last token representation as chunk embedding
        emb = hidden[:, -1, :].squeeze(0)  # (D)

        g_new = self.global_update(emb.unsqueeze(0), prev_global.unsqueeze(0)).squeeze(0)
        if character_present:
            c_new = self.char_update(emb.unsqueeze(0), prev_char.unsqueeze(0)).squeeze(0)
        else:
            c_new = prev_char

        proj_emb = self.proj(emb)
        return g_new, c_new, proj_emb

    @torch.no_grad()
    def embed_claim(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text).unsqueeze(0).to(self.device)
        # Truncate to max 256 tokens for claims
        if ids.shape[1] > 256:
            ids = ids[:, :256]
        hidden = self.core.encode(ids)
        emb = hidden[:, -1, :].squeeze(0)
        return self.proj(emb)
