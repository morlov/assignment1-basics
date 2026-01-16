import torch
from torch import Tensor
from jaxtyping import Float, Bool
import einops
import math

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    exp_along_dim = torch.exp(in_features - in_features.max(dim=dim, keepdim=True).values)
    sum_along_dim = torch.sum(exp_along_dim, dim=dim, keepdim=True)
    return exp_along_dim/sum_along_dim

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:

    d_k = Q.size(-1)

    scores = einops.einsum(Q, K, "... q d_k, ... k d_k -> ... q k")/math.sqrt(d_k)
    
    if mask is not None:
        scores = scores + torch.where(mask, 0.0, float("-inf"))

    weights = softmax(scores, dim=-1)
    output = einops.einsum(weights, V, "... q k, ... k d_v -> ... q d_v")

    return output

