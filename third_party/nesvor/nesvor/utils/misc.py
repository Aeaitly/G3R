from typing import Optional, Union, Collection, Sequence
import collections
import torch
import torch.nn.functional as F
from .types import DeviceType


def resample(
    x: torch.Tensor, res_xyz_old: Sequence, res_xyz_new: Sequence
) -> torch.Tensor:
    ndim = x.ndim - 2
    assert len(res_xyz_new) == len(res_xyz_old) == ndim
    if all(r_new == r_old for (r_new, r_old) in zip(res_xyz_new, res_xyz_old)):
        return x
    grids = []
    for i in range(ndim):
        fac = res_xyz_old[i] / res_xyz_new[i]
        size_new = int(x.shape[-i - 1] * fac)
        grid_max = (size_new - 1) / fac / (x.shape[-i - 1] - 1)
        grids.append(
            torch.linspace(
                -grid_max, grid_max, size_new, dtype=x.dtype, device=x.device
            )
        )
    grid = torch.stack(torch.meshgrid(*grids[::-1], indexing="ij")[::-1], -1)
    y = F.grid_sample(
        x, grid[None].expand((x.shape[0],) + (-1,) * (ndim + 1)), align_corners=True
    )
    return y

def meshgrid(
    shape_xyz: Collection,
    resolution_xyz: Collection,
    min_xyz: Optional[Collection] = None,
    device: DeviceType = None,
    stack_output: bool = True,
):
    assert len(shape_xyz) == len(resolution_xyz)
    if min_xyz is None:
        min_xyz = tuple(-(s - 1) * r / 2 for s, r in zip(shape_xyz, resolution_xyz))
    else:
        assert len(shape_xyz) == len(min_xyz)

    if device is None:
        if isinstance(shape_xyz, torch.Tensor):
            device = shape_xyz.device
        elif isinstance(resolution_xyz, torch.Tensor):
            device = resolution_xyz.device
        else:
            device = torch.device("cpu")
    dtype = torch.float32

    arr_xyz = [
        torch.arange(s, dtype=dtype, device=device) * r + m
        for s, r, m in zip(shape_xyz, resolution_xyz, min_xyz)
    ]
    grid_xyz = torch.meshgrid(arr_xyz[::-1], indexing="ij")[::-1]
    if stack_output:
        return torch.stack(grid_xyz, -1)
    else:
        return grid_xyz

def gaussian_blur(
    x: torch.Tensor, sigma: Union[float, collections.abc.Iterable], truncated: float
) -> torch.Tensor:
    spatial_dims = len(x.shape) - 2
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = [sigma] * spatial_dims
    kernels = [gaussian_1d_kernel(s, truncated, x.device) for s in sigma]
    c = x.shape[1]
    conv_fn = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
    for d in range(spatial_dims):
        s = [1] * len(x.shape)
        s[d + 2] = -1
        k = kernels[d].reshape(s).repeat(*([c, 1] + [1] * spatial_dims))
        padding = [0] * spatial_dims
        padding[d] = (k.shape[d + 2] - 1) // 2
        x = conv_fn(x, k, padding=padding, groups=c)
    return x

def gaussian_1d_kernel(
    sigma: float, truncated: float, device: DeviceType
) -> torch.Tensor:
    tail = int(max(sigma * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
    t = 0.70710678 / sigma
    kernel = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    return kernel.clamp(min=0)
