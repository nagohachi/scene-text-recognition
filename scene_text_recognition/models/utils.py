import torch


def lens_to_mask(lens: torch.Tensor) -> torch.Tensor:
    """generate mask from lens.

    Args:
        lens (torch.Tensor[int]): lengths tensor: (batch_size, )

    Returns:
        torch.Tensor: mask tensor.

    Examples:
    >>> lens = torch.tensor([1, 2, 5, 3], dtype=torch.long)
    >>> lens_to_mask(lens)
    tensor([[False,  True,  True,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False],
            [False, False, False,  True,  True]])
    """
    max_length = lens.max().item()
    indices = torch.arange(max_length, device=lens.device)
    return indices >= lens[:, None]
