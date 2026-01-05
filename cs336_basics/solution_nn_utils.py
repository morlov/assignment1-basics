import torch
from torch import Tensor
from jaxtyping import Float

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