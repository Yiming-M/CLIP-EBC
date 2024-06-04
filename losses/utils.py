from torch import Tensor


def _reshape_density(density: Tensor, reduction: int) -> Tensor:
    assert len(density.shape) == 4, f"Expected 4D (B, 1, H, W) tensor, got {density.shape}"
    assert density.shape[1] == 1, f"Expected 1 channel, got {density.shape[1]}"
    assert density.shape[2] % reduction == 0, f"Expected height to be divisible by {reduction}, got {density.shape[2]}"
    assert density.shape[3] % reduction == 0, f"Expected width to be divisible by {reduction}, got {density.shape[3]}"
    return density.reshape(density.shape[0], 1, density.shape[2] // reduction, reduction, density.shape[3] // reduction, reduction).sum(dim=(-1, -3))
