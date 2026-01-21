from typing import Dict
import math

import torch
from torch import Tensor
from typing import Optional
from einops import einsum, rearrange

from .solution_nn_utils import scaled_dot_product_attention, softmax


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None): 
        """
        Construct a linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
        self._reset_parameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.weight, "... d_in, d_out d_in-> ... d_out")

    def _reset_parameters(self):
        std = math.sqrt(2/(self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight , mean=0, std=std, a=-3*std, b=-3*std)


class Embedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        
        super().__init__()
        
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            )
        
        self._reset_parameters()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        indices  = rearrange(token_ids, "batch ... -> (batch ...)")
        selected = torch.index_select(self.weight, 0, indices)
        batch, seq = token_ids.size(0), token_ids.size(1)
        return rearrange(selected, "(batch seq) dim -> batch seq dim", batch=batch, seq=seq)

    def _reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)


class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):

        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        self._reset_parameters()


    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = einsum(x.pow(2),  "... d -> ...")/self.d_model 
        rms = rearrange(torch.sqrt(var + self.eps), "... -> ... 1")
        rms_norm = x/rms * self.weight
        return rms_norm.to(in_dtype) 

    def _reset_parameters(self):
        """Initialize parameters according to assignment specification."""
        torch.nn.init.ones_(self.weight)


class SiLU(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class SwigLUFFN(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):

        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x):
        """
            W2(SiLU(W1x) ⊙ W3x)
        """
        w1x = self.w1(x)
        w3x = self.w3(x) 
        z = self.silu(w1x) * w3x
        return self.w2(z)


class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        
        """
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """

        super().__init__()

        assert d_k%2 == 0, "Dimension of d_k should be divisible by 2"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        i = torch.arange(self.max_seq_len, dtype=torch.float32).to(self.device)
        k = torch.arange(self.d_k//2, dtype=torch.float32).to(self.device)
        
        inv_thetas = 1/(self.theta ** (2*k/self.d_k))

        thetas = torch.outer(i, inv_thetas)

        sin_vals = torch.sin(thetas)
        cos_vals = torch.cos(thetas)

        self.register_buffer("sin", sin_vals, persistent=False)
        self.register_buffer("cos", cos_vals, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
            Process an input tensor of shape (..., seq_len, d_k) 
            and return a tensor of the same shape.
        """

        x_odd = x[..., 1::2]
        x_even = x[..., ::2]

        sin = self.sin[token_positions]
        cos = self.cos[token_positions] 

        x_rotated_odd = x_even * sin + x_odd * cos
        x_rotated_even = x_even * cos - x_odd * sin

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated


class CausalMultiHeadSelfAttention(torch.nn.Module):

    def __init__(
        self, d_model: int, 
        num_heads: int, 
        max_seq_len: int = 2048,
        use_rope: bool = False,
        theta: float = 10000, 
        device = None, 
        dtype = None
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        
        assert d_model%num_heads==0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = self.d_model//num_heads

        self.q_proj = Linear(d_model, self.d_k*num_heads, device=self.device, dtype=self.dtype)
        self.k_proj = Linear(d_model, self.d_k*num_heads, device=self.device, dtype=self.dtype)
        self.v_proj = Linear(d_model, self.d_v*num_heads, device=self.device, dtype=self.dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device=self.device, dtype=self.dtype)

        self.use_rope = use_rope

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, 
                d_k=self.d_k, 
                max_seq_len=max_seq_len, 
                device=device
            )

        self.register_buffer(
            'attn_mask', 
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),diagonal=1),
            persistent=False
            )

    def forward(self, x: torch.Tensor, token_positions = None):
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(
            q,
            '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', 
            num_heads=self.num_heads,
            d_k=self.d_k
        )
        
        k = rearrange(
            k,
            '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k',  
            num_heads=self.num_heads,
            d_k=self.d_k
        )
        v = rearrange(
            v,
            '... seq_len (num_heads d_v) -> ... num_heads seq_len d_v',  
            num_heads=self.num_heads,
            d_v=self.d_v
        )

        seq_len = q.shape[-2]

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = ~self.attn_mask[:seq_len, :seq_len]
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        attn_output = rearrange(
            attn_output,
            '... num_heads seq_len d_k -> ... seq_len (num_heads d_k)', 
            num_heads=self.num_heads,
            d_k = self.d_k
        )

        return self.output_proj(attn_output)

class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device = None,
        dtype = None
    ):
        super().__init__()
        
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            use_rope=True, 
            theta=theta,
            device=device,
            dtype=dtype)

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwigLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)


    def forward(self, x: Tensor, token_positions: Optional[torch.Tensor] = None):
        
        z = self.ln1(x)
        z = self.attn(z, token_positions)
        x = x + z
        
        z = self.ln2(x)
        z = self.ffn(z)
        x = x + z

        return x

class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
        device: None,
        dtype: None
    ):

        super().__init__()

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            device=device,
            dtype=dtype
        )

        self.layers = torch.nn.ModuleList([
                TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices):
        
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


