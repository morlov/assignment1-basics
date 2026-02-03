import numpy as np
import numpy.typing as npt
import torch


from einops import rearrange

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    
    max_index = len(dataset) - context_length

    inputs, targets = [], []
    for i in range(batch_size):
        sampled_id = np.random.randint(0, max_index)
        subsample = dataset[sampled_id:(sampled_id + context_length + 1)]
        inputs.append(subsample[:-1])
        targets.append(subsample[1:])

    inputs = torch.tensor(inputs, dtype=torch.int32, device=device)
    targets = torch.tensor(targets, dtype=torch.int32, device=device)

    return (inputs, targets)