import torch
from einops import einsum, rearrange
import math


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

        self.weights = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
        self._reset_parameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.weights, "... d_in, d_out d_in-> ... d_out")

    def _reset_parameters(self):
        std = math.sqrt(2/(self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weights , mean=0, std=std, a=-3*std, b=-3*std)


class Embedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        
        super().__init__()
        
        self.weights = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            )
        
        self._reset_parameters()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        indices  = rearrange(token_ids, "batch ... -> (batch ...)")
        selected = torch.index_select(self.weights, 0, indices)
        batch, seq = token_ids.size(0), token_ids.size(1)
        return rearrange(selected, "(batch seq) dim -> batch seq dim", batch=batch, seq=seq)

    def _reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weights , mean=0, std=1, a=-3, b=3)


class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):

        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.weights = torch.nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        self._reset_parameters()


    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = einsum(x.pow(2),  "... d -> ...")/self.d_model 
        rms = rearrange(torch.sqrt(var + self.eps), "... -> ... 1")
        rms_norm = x/rms * self.weights
        return rms_norm.to(in_dtype) 

    def _reset_parameters(self):
        """Initialize parameters according to assignment specification."""
        torch.nn.init.ones_(self.weights)


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

        x_rotated_odd = x_even * self.sin + x_odd * self.cos
        x_rotated_even = x_even * self.cos - x_odd * self.sin

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated



