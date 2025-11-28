import torch
from torch import nn
from typing import Optional
from fla.layers.rwkv6 import RWKV6Attention

class RwkvEncoder(nn.Module):
    def __init__(self, 
                hidden_size: int = 1024,
                expand_k: float = 0.5,
                expand_v: float = 1.0,
                num_heads: int = 4,
                gate_fn: str = 'swish',
                proj_low_rank_dim: int = 32,
                gate_low_rank_dim: int = 64,
                elementwise_affine: Optional[bool] = True,
                norm_eps: float = 1e-5,
                layer_idx: int = None,
                ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.gate_fn = gate_fn
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.layer_idx = layer_idx

        retnet = RWKV6Attention(
            mode="chunk",
            hidden_size=self.hidden_size,
            expand_k=self.expand_k,
            expand_v=self.expand_v,
            num_heads=self.num_heads,
            gate_fn=self.gate_fn,
            proj_low_rank_dim=self.proj_low_rank_dim,
            gate_low_rank_dim=self.gate_low_rank_dim,
            elementwise_affine=self.elementwise_affine,
            norm_eps=self.norm_eps,
            layer_idx=self.layer_idx,
        )
        self.net = retnet

    def forward(self, 
                x: torch.Tensor, 
                past_key_values: Optional[torch.Tensor] = None, 
                use_cache: bool = False):
        
        # Check the internal PyTorch flag 'self.training'
        # This flag is automatically toggled when you call model.train() or model.eval()
        if self.training:
            self.net.mode = 'chunk'
            output, _ = self.net(x)
            return output
        else:
            # In Eval mode, switch to recurrent for O(1) generation
            self.net.mode = 'fused_recurrent'
            
            output, current_key_values = self.net(
                x, 
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            
            if use_cache:
                return output, current_key_values
            return output
