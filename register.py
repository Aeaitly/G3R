"""Preprocess stacks and infer slice poses with the G3R pretrained model."""

import logging
from pathlib import Path
import sys
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "nesvor"))

import numpy as np
import torch
import torch.nn.functional as F
from nesvor.svort import G3RPriorV1, G3RPriorV2
from nesvor.transform import RigidTransform
from nesvor.utils import get_PSF, resample
from nesvor.image import Stack, Slice, Volume


def parse_data(dataset: List[Stack], svort: bool) -> Tuple[
    List[Stack],
    List[Stack],
    List[Stack],
    List[Stack],
    List[torch.Tensor],
    Volume,
    torch.Tensor,
]:
    stacks = []
    stacks_ori = []
    transforms = []
    transforms_full = []
    transforms_ori = []
    crop_idx = []
    dataset_out = []
    res_s = 1.0
    res_r = 0.8
    for data in dataset:
        logging.debug("Preprocessing stack %s for registration.", data.name)
        slices = resample(
            data.slices * data.mask,
            (data.resolution_x, data.resolution_y),
            (res_s, res_s),
        )
        slices_ori = slices.clone()
        s = slices[torch.argmax((slices > 0).sum((1, 2, 3))), 0]
        i1, i2, j1, j2 = (0, s.shape[0] - 1, 0, s.shape[1] - 1)
        while i1 < s.shape[0] and s[i1, :].sum() == 0:
            i1 += 1
        while i2 and s[i2, :].sum() == 0:
            i2 -= 1
        while j1 < s.shape[1] and s[:, j1].sum() == 0:
            j1 += 1
        while j2 and s[:, j2].sum() == 0:
            j2 -= 1
        if (i2 - i1 > 128 or j2 - j1 > 128) and svort:
            logging.warning('ROI in input stack "%s" is too large for the G3R pretrained model', data.name)
        if i2 - i1 <= 0:
            logging.warning(
                'Input stack "%s" is all zero after maksing and will be skipped. Please check your data!',
                data.name,
            )
            continue
        pad_margin = 64
        slices = F.pad(
            slices, (pad_margin, pad_margin, pad_margin, pad_margin), "constant", 0
        )
        i = pad_margin + (i1 + i2) // 2
        j = pad_margin + (j1 + j2) // 2
        slices = slices[:, :, i - 64 : i + 64, j - 64 : j + 64]
        idx = (slices > 0).float().sum((1, 2, 3)) > 0
        nz = torch.nonzero(idx)
        nnz = torch.numel(nz)
        if nnz < 7:
            logging.warning(
                'Input stack "%s" only has %d nonzero slices after masking. Consider remove this stack.',
                data.name,
                nnz,
            )
        else:
            logging.debug(
                'Input stack "%s" has %d nonzero slices after masking.', data.name, nnz
            )
        idx[int(nz[0, 0]) : int(nz[-1, 0] + 1)] = True
        crop_idx.append(idx)
        slices = slices[idx]
        stacks.append(slices / torch.quantile(slices[slices > 0], 0.99))
        stacks_ori.append(slices_ori)
        transform = data.transformation
        transforms_ori.append(transform)
        transform_full_ax = transform.axisangle().clone()
        transform_ax = transform_full_ax[idx].clone()
        transform_full_ax[:, :-1] = 0
        transform_full_ax[:, 3] = -((j1 + j2) // 2 - slices_ori.shape[-1] / 2) * res_s
        transform_full_ax[:, 4] = -((i1 + i2) // 2 - slices_ori.shape[-2] / 2) * res_s
        transform_full_ax[:, -1] -= transform_ax[:, -1].mean()
        transform_ax[:, :-1] = 0
        transform_ax[:, -1] -= transform_ax[:, -1].mean()
        transforms.append(RigidTransform(transform_ax))
        transforms_full.append(RigidTransform(transform_full_ax))
        dataset_out.append(data)
    assert len(dataset_out) > 0, "Input data is empty!"
    s_thick = np.mean([data.thickness for data in dataset_out])
    gaps = [data.gap for data in dataset_out]
    stacks_svort_in = [
        Stack(stacks[j], stacks[j] > 0, transforms[j], res_s, res_s, s_thick, gaps[j])
        for j in range(len(dataset_out))
    ]
    stacks_resampled = [
        Stack(
            stacks_ori[j],
            stacks_ori[j] > 0,
            transforms_ori[j],
            res_s,
            res_s,
            s_thick,
            gaps[j],
        )
        for j in range(len(dataset_out))
    ]
    stacks_resampled_reset = [s.clone(zero=False, deep=False) for s in stacks_resampled]
    for j in range(len(dataset_out)):
        stacks_resampled_reset[j].transformation = transforms_full[j]
    volume = Volume.zeros((200, 200, 200), res_r, device=dataset_out[0].device)
    psf = get_PSF(
        res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r), device=volume.device
    )
    return (
        dataset_out,
        stacks_svort_in,
        stacks_resampled,
        stacks_resampled_reset,
        crop_idx,
        volume,
        psf,
    )


def _register(args, dataset):
    svort = args.use_svort
    res_r = args.resolution_volume
    device_index = args.device
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    models = {"v1": G3RPriorV1, "v2": G3RPriorV2}
    print(f"Using the G3R pretrained model {args.model_variant}.")
    model = models[args.model_variant](n_iter=4)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    dataset, stacks_in, ss_ori, stacks_full, crop_idx, volume, psf = parse_data(
        dataset, svort
    )
    origin_device = stacks_in[0].device
    positions = torch.cat(
        [
            torch.stack(
                (
                    torch.arange(len(s), dtype=torch.float32, device=device)
                    - len(s) // 2,
                    torch.full((len(s),), i, dtype=torch.float32, device=device),
                ),
                dim=-1,
            )
            for i, s in enumerate(stacks_in)
        ],
        dim=0,
    )
    res_s = stacks_in[0].resolution_x
    data = {
        "psf_rec": psf.to(device=device),
        "slice_shape": stacks_in[0].shape[-2:],
        "resolution_slice": res_s,
        "resolution_recon": res_r,
        "volume_shape": volume.shape,
        "transforms": RigidTransform.cat([s.transformation for s in stacks_in])
        .matrix()
        .to(device),
        "stacks": torch.cat([s.slices for s in stacks_in], dim=0).to(device),
        "positions": positions,
        "slice_thickness": stacks_in[0].thickness,
    }
    with torch.no_grad():
        t_out, v_out, _ = model(data)
    transforms_out = [t_out[-1][positions[:, -1] == i] for i in range(len(stacks_in))]
    stacks_out = []
    for i in range(len(stacks_in)):
        stack_out = stacks_in[i].clone(zero=False, deep=False)
        stack_out.transformation = transforms_out[i].to(origin_device)
        stack_out.slices = stack_out.slices.to(origin_device)
        stack_out.mask = stack_out.mask.to(origin_device)
        stacks_out.append(stack_out)
    slices = []
    for stack in stacks_out:
        idx_nonempty = stack.mask.flatten(1).any(1)
        stack.slices /= torch.quantile(stack.slices[stack.mask], 0.99)
        slices.extend(stack[idx_nonempty])
    return (slices, v_out[-1].to(origin_device), stacks_out)
