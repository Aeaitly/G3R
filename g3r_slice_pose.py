"""G3R reconstruction using the pretrained model and slice poses."""
# G3R modifications to the Apache-2.0 GSVR implementation. See LICENSE and NOTICE.

import math
import os
import random
import time
import warnings
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "nesvor"))
import faiss
import faiss.contrib.torch_utils
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.data import Subject
from register import _register

warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")
warnings.filterwarnings(
    "ignore", message=".*Please use the new API settings to control TF32.*"
)
torch.set_float32_matmul_precision("high")


def fused_mahalanobis_distance(sigmas_nk, x_minus_mu_nk):
    """Evaluate the 3D quadratic form using an explicit covariance inverse."""
    a = sigmas_nk[..., 0, 0]
    b = sigmas_nk[..., 0, 1]
    c = sigmas_nk[..., 0, 2]
    e = sigmas_nk[..., 1, 1]
    f = sigmas_nk[..., 1, 2]
    h = sigmas_nk[..., 2, 1]
    i = sigmas_nk[..., 2, 2]
    det = a * (e * i - f * h) - b * (b * i - f * c) + c * (b * h - e * c)
    inv_det = 1.0 / (det + 1e-08)
    inv_00 = (e * i - f * h) * inv_det
    inv_01 = (c * h - b * i) * inv_det
    inv_02 = (b * f - c * e) * inv_det
    inv_10 = inv_01
    inv_11 = (a * i - c * c) * inv_det
    inv_12 = (c * b - a * f) * inv_det
    inv_20 = inv_02
    inv_21 = inv_12
    inv_22 = (a * e - b * b) * inv_det
    vx = x_minus_mu_nk[..., 0]
    vy = x_minus_mu_nk[..., 1]
    vz = x_minus_mu_nk[..., 2]
    t0 = inv_00 * vx + inv_01 * vy + inv_02 * vz
    t1 = inv_10 * vx + inv_11 * vy + inv_12 * vz
    t2 = inv_20 * vx + inv_21 * vy + inv_22 * vz
    result = vx * t0 + vy * t1 + vz * t2
    return result


def fused_motion_correction_kernel(
    coords_n_3,
    quats_k_4,
    trans_k_3,
    slice_ids_n,
    slice_centers_n_3,
    sigma_psf_n_3_3=None,
):
    q_w = quats_k_4[slice_ids_n, 0]
    q_x = quats_k_4[slice_ids_n, 1]
    q_y = quats_k_4[slice_ids_n, 2]
    q_z = quats_k_4[slice_ids_n, 3]
    t_x = trans_k_3[slice_ids_n, 0]
    t_y = trans_k_3[slice_ids_n, 1]
    t_z = trans_k_3[slice_ids_n, 2]
    inv_norm = torch.rsqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z + 1e-08)
    w = q_w * inv_norm
    x = q_x * inv_norm
    y = q_y * inv_norm
    z = q_z * inv_norm
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    r00 = 1.0 - 2.0 * (y2 + z2)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (x2 + z2)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (x2 + y2)
    c_x = coords_n_3[:, 0] - slice_centers_n_3[:, 0]
    c_y = coords_n_3[:, 1] - slice_centers_n_3[:, 1]
    c_z = coords_n_3[:, 2] - slice_centers_n_3[:, 2]
    rot_x = r00 * c_x + r01 * c_y + r02 * c_z
    rot_y = r10 * c_x + r11 * c_y + r12 * c_z
    rot_z = r20 * c_x + r21 * c_y + r22 * c_z
    final_x = rot_x + t_x + slice_centers_n_3[:, 0]
    final_y = rot_y + t_y + slice_centers_n_3[:, 1]
    final_z = rot_z + t_z + slice_centers_n_3[:, 2]
    coords_mc = torch.stack([final_x, final_y, final_z], dim=-1)
    sigma_tf = None
    if sigma_psf_n_3_3 is not None:
        s00 = sigma_psf_n_3_3[:, 0, 0]
        s01 = sigma_psf_n_3_3[:, 0, 1]
        s02 = sigma_psf_n_3_3[:, 0, 2]
        s10 = sigma_psf_n_3_3[:, 1, 0]
        s11 = sigma_psf_n_3_3[:, 1, 1]
        s12 = sigma_psf_n_3_3[:, 1, 2]
        s20 = sigma_psf_n_3_3[:, 2, 0]
        s21 = sigma_psf_n_3_3[:, 2, 1]
        s22 = sigma_psf_n_3_3[:, 2, 2]
        t00 = r00 * s00 + r01 * s10 + r02 * s20
        t01 = r00 * s01 + r01 * s11 + r02 * s21
        t02 = r00 * s02 + r01 * s12 + r02 * s22
        t10 = r10 * s00 + r11 * s10 + r12 * s20
        t11 = r10 * s01 + r11 * s11 + r12 * s21
        t12 = r10 * s02 + r11 * s12 + r12 * s22
        t20 = r20 * s00 + r21 * s10 + r22 * s20
        t21 = r20 * s01 + r21 * s11 + r22 * s21
        t22 = r20 * s02 + r21 * s12 + r22 * s22
        res00 = t00 * r00 + t01 * r01 + t02 * r02
        res01 = t00 * r10 + t01 * r11 + t02 * r12
        res02 = t00 * r20 + t01 * r21 + t02 * r22
        res10 = t10 * r00 + t11 * r01 + t12 * r02
        res11 = t10 * r10 + t11 * r11 + t12 * r12
        res12 = t10 * r20 + t11 * r21 + t12 * r22
        res20 = t20 * r00 + t21 * r01 + t22 * r02
        res21 = t20 * r10 + t21 * r11 + t22 * r12
        res22 = t20 * r20 + t21 * r21 + t22 * r22
        sigma_tf = torch.stack(
            [
                torch.stack([res00, res01, res02], dim=-1),
                torch.stack([res10, res11, res12], dim=-1),
                torch.stack([res20, res21, res22], dim=-1),
            ],
            dim=1,
        )
    return (coords_mc, sigma_tf)


class GSVR(nn.Module):
    """Anisotropic Gaussian field with per-slice motion and intensity variables."""

    def __init__(
        self,
        num_gaussians,
        num_slices,
        D,
        bbox_wrld,
        mc=False,
        apply_slice_scaling=False,
        apply_slice_uncertainty=False,
    ):
        super().__init__()
        self.D = 3
        self.bbox_wrld = bbox_wrld
        self.mc = mc
        if D != 3:
            print(
                f"Warning: Initializing 3D Gaussian Primitives SVR model but D was set to {D}. Forcing D=3."
            )
        self.num_gaussians = num_gaussians
        self.sigmoid = nn.Sigmoid()
        self.mu = nn.Parameter(
            (torch.rand(num_gaussians, self.D) * 2 - 1) * (bbox_wrld.cpu() // 2 - 1)
        )
        self.scaling = nn.Parameter(
            (
                torch.ones(num_gaussians, self.D)
                + torch.randn(num_gaussians, self.D) * 0.1
            )
            * 0.1
        )
        self.rotation_g = nn.Parameter(torch.zeros(num_gaussians, 4))
        self.rotation_g.data[:, 0] = 1.0
        self.color = nn.Parameter(torch.ones(num_gaussians) * 0.5)
        self.rotation_mc = nn.Parameter(torch.zeros(num_slices, 4)) if mc else None
        if mc:
            self.rotation_mc.data[:, 0] = 1.0
        self.translation_mc = nn.Parameter(torch.zeros(num_slices, 3)) if mc else None
        if apply_slice_scaling:
            self.slice_scaling = nn.Parameter(torch.ones(num_slices))
        if apply_slice_uncertainty:
            self.slice_weight = nn.Parameter(torch.ones(num_slices))

    def initialize_parameters_from_svortvolume_new(
        self,
        volume_np,
        volume_affine,
        device=None,
        intensity_thresh=0.05,
        max_voxels=500000,
        init_scale_mm=1.5,
    ):
        print("[Init] Initializing Gaussian positions from the pretrained model prior")
        assert volume_np.ndim == 3, "volume must be (D, H, W)"
        vmax = float(volume_np.max())
        if vmax <= 0:
            raise ValueError("The pretrained model prior is empty")
        mask = volume_np > intensity_thresh * vmax
        coords_zyx = np.argwhere(mask)
        values = volume_np[mask]
        if len(coords_zyx) == 0:
            raise ValueError("No foreground voxels in the pretrained model prior")
        if len(coords_zyx) > max_voxels:
            idx = np.random.choice(len(coords_zyx), max_voxels, replace=False)
            coords_zyx = coords_zyx[idx]
            values = values[idx]
        coords_xyz = coords_zyx[:, [2, 1, 0]]
        ones = np.ones((coords_xyz.shape[0], 1), dtype=np.float32)
        coords_h = np.concatenate([coords_xyz, ones], axis=1)
        coords_world = (volume_affine @ coords_h.T).T[:, :3]
        if coords_world.shape[0] < self.num_gaussians:
            print(
                f"[Init:Warn] Foreground voxels ({coords_world.shape[0]}) < num_gaussians. Enabling replacement."
            )
            idx = np.random.choice(
                coords_world.shape[0], self.num_gaussians, replace=True
            )
        else:
            idx = np.random.choice(
                coords_world.shape[0], self.num_gaussians, replace=False
            )
        mu_init = torch.from_numpy(coords_world[idx]).to(
            dtype=torch.float32, device=device
        )
        color_vals = (
            torch.from_numpy(values[idx])
            .to(dtype=torch.float32, device=device)
            .unsqueeze(1)
        )
        with torch.no_grad():
            self.mu.data.copy_(mu_init)
            self.scaling.data.fill_(math.log(init_scale_mm))
            color_vals_flat = color_vals.view(-1)
            if self.color.ndim == 1:
                self.color.data.copy_(color_vals_flat)
            else:
                color_vals = color_vals.unsqueeze(1)
                C = self.color.shape[1]
                if C > 1:
                    color_vals = color_vals.repeat(1, C)
                self.color.data.copy_(color_vals)
        self.scaling.requires_grad = True
        self.color.requires_grad = True
        print(
            f"[Init] Initialized {self.num_gaussians} Gaussians\n           scale = {init_scale_mm:.2f} mm (learnable)\n           color = initialized from volume intensity (learnable)"
        )

    def forward(
        self,
        coords,
        slice_ids=None,
        slice_centers=None,
        Sigma_psf=None,
        top_k_idcs=None,
        scale_scale=1.0,
        scale_threshold=1.0,
    ):
        q = None
        t = None
        mus = self.mu[top_k_idcs] if top_k_idcs is not None else self.mu
        x_minus_mu = coords.unsqueeze(1) - mus
        sigmas = self.compute_sigma(scale_scale)
        sigmas = sigmas[top_k_idcs]
        if Sigma_psf is not None:
            sigmas = (sigmas + Sigma_psf[:, None, :, :]).to(dtype=torch.float32)
        else:
            sigmas = sigmas.to(dtype=torch.float32)
        inner_term = fused_mahalanobis_distance(sigmas, x_minus_mu)
        spatial_weights = torch.exp(-0.5 * inner_term)
        color_vals = self.color[top_k_idcs]
        raw_weights = spatial_weights
        # Preserve the original soft normalization of overlapping Gaussians.
        delta = 0.05
        sum_weights = torch.sum(raw_weights, dim=1, keepdim=True)
        gaussian_weights_nrmd = raw_weights / (sum_weights + delta)
        intensity = torch.sum(gaussian_weights_nrmd * color_vals, dim=1)
        if scale_scale != 1.0:
            intensity = intensity * (sum_weights.squeeze(-1) > scale_threshold).float()
        return (intensity, q, t)

    def motion_correction_fused(self, coords, slice_ids, slice_centers, Sigma_psf=None):
        quats = self.rotation_mc
        trans = self.translation_mc
        coords_mc, Sigma_psf_tf = fused_motion_correction_kernel(
            coords, quats, trans, slice_ids, slice_centers, Sigma_psf
        )
        batch_q = quats[slice_ids]
        batch_t = trans[slice_ids]
        rot = batch_q[:, 1:]
        return (coords_mc, Sigma_psf_tf, rot, batch_t)

    def build_rotation_matrix_from_quaternion(self, q):
        q_norm = torch.nn.functional.normalize(q, p=2, dim=1)
        w, x, y, z = (q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3])
        K = q.shape[0]
        R = torch.empty((K, 3, 3), device=q.device, dtype=q.dtype)
        x2, y2, z2 = (x * x, y * y, z * z)
        xy, xz, yz = (x * y, x * z, y * z)
        wx, wy, wz = (w * x, w * y, w * z)
        R[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
        R[:, 0, 1] = 2.0 * (xy - wz)
        R[:, 0, 2] = 2.0 * (xz + wy)
        R[:, 1, 0] = 2.0 * (xy + wz)
        R[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
        R[:, 1, 2] = 2.0 * (yz - wx)
        R[:, 2, 0] = 2.0 * (xz - wy)
        R[:, 2, 1] = 2.0 * (yz + wx)
        R[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)
        return R

    def compute_sigma(self, scale_scale=1.0):
        R = self.build_rotation_matrix_from_quaternion(self.rotation_g)
        scaling = torch.exp(self.scaling)
        S = torch.diag_embed(scaling * scale_scale)
        Sigma = torch.bmm(R, torch.bmm(S, R.transpose(1, 2)))
        return Sigma


def simple_knn_torch(queries, data_points, k, chunk_size=5000):
    N = queries.shape[0]
    indices_list = []
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        batch_queries = queries[i:end]
        dists = torch.cdist(batch_queries, data_points)
        _, idx = torch.topk(dists, k, dim=1, largest=False)
        indices_list.append(idx)
    return torch.cat(indices_list, dim=0)


def create_vis_grid(grid_shape, bbox_wrld, device):
    coords_axes = [
        torch.linspace(-1, 1, s) * (b / 2) for s, b in zip(grid_shape, bbox_wrld)
    ]
    grid_x, grid_y, grid_z = torch.meshgrid(*coords_axes, indexing="ij")
    coords_flat = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1
    )
    coords_flat = coords_flat.to(device)
    return coords_flat


def visualize_gaussians(
    groundtruth,
    gs,
    gpu_index,
    vis_grid_flat,
    shape_rec,
    spacing_rec,
    new_origin,
    epoch=0,
    k=10,
    BATCH_SIZE=100000,
    psf_scale_fac=1.0,
    output_file_path=None,
    PSF=False,
    ttime=0,
    min_value=0.0,
    max_value=1.0,
    scale_scale=0.0,
    scale_threshold=1.0,
):
    print("BatchSize:", BATCH_SIZE)
    device = vis_grid_flat.device
    values_pred_flat = torch.empty((vis_grid_flat.shape[0],), device=device)
    for step in range(0, vis_grid_flat.shape[0], BATCH_SIZE):
        with torch.no_grad():
            _, top_k_idcs = gpu_index.search(vis_grid_flat[step : step + BATCH_SIZE], k)
            if (top_k_idcs >= gs.num_gaussians).any() or (top_k_idcs < 0).any():
                print("Error: Faiss returned invalid indices!")
                top_k_idcs = torch.clamp(top_k_idcs, min=0, max=gs.num_gaussians - 1)
            Sigma_psf = (
                generate_cov_psf(
                    np.eye(3),
                    spacing_rec,
                    slice_thickness=None,
                    scale_factor=psf_scale_fac,
                )
                if PSF
                else None
            )
            Sigma_psf = (
                torch.tensor(np.broadcast_to(Sigma_psf, (len(top_k_idcs), 3, 3)))
                .float()
                .to(device)
                if Sigma_psf is not None
                else None
            )
            values_pred_flat[step : step + BATCH_SIZE] = gs(
                vis_grid_flat[step : step + BATCH_SIZE],
                slice_ids=None,
                slice_centers=None,
                Sigma_psf=Sigma_psf,
                top_k_idcs=top_k_idcs,
                scale_scale=scale_scale,
                scale_threshold=scale_threshold,
            )[0]
    values_pred = (
        values_pred_flat.reshape(shape_rec).detach().cpu().numpy().clip(0.0, None)
    )
    values_pred = values_pred * (max_value - min_value) + min_value
    groundtruth_data = groundtruth.get_fdata() if groundtruth is not None else None
    rec_affine = np.diag(list(spacing_rec) + [1.0])
    rec_affine[:3, 3] = new_origin
    rec_nii = nib.Nifti1Image(values_pred, rec_affine)
    filename = output_file_path.replace(".nii.gz", f"_ttime={ttime:.2f}s.nii.gz")
    nib.save(rec_nii, filename)
    if epoch == 0 and groundtruth is not None:
        groundtruth_3D = nib.Nifti1Image(groundtruth_data, groundtruth.affine)
        nib.save(
            groundtruth_3D,
            output_file_path.replace(
                ".nii.gz", f"_groundtruth_3D_epoch_{epoch}.nii.gz"
            ),
        )
    torch.cuda.empty_cache()
    return values_pred


def _init_optim(gs, MC, SLICE_SCALING, SLICE_UNCERTAINTY, max_epochs, lrs):
    param_groups = [
        {"params": gs.mu, "lr": lrs["position"], "name": "gs_mu"},
        {"params": gs.scaling, "lr": lrs["scaling"], "name": "gs_scaling"},
        {"params": gs.rotation_g, "lr": lrs["rotation"], "name": "gs_rotation_g"},
        {"params": gs.color, "lr": lrs["color"], "name": "gs_color"},
    ]
    if MC:
        param_groups.append({"params": gs.rotation_mc, "lr": lrs["motion_rot"]})
        param_groups.append({"params": gs.translation_mc, "lr": lrs["motion_trans"]})
    if SLICE_SCALING:
        param_groups.append(
            {
                "params": gs.slice_scaling,
                "lr": lrs["slice_scale"],
                "name": "gs_slice_scaling",
            }
        )
    if SLICE_UNCERTAINTY:
        param_groups.append(
            {
                "params": gs.slice_weight,
                "lr": lrs["slice_weight"],
                "name": "gs_slice_weights",
            }
        )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    from torch.cuda.amp import GradScaler

    grad_scaler = GradScaler()
    return (optimizer, scheduler, grad_scaler)


def resample_svort_to_gsvr_space(
    svort_volume_np, svort_affine, shape_rec, spacing_rec, new_origin, device="cuda"
):
    sx, sy, sz = spacing_rec
    x0, y0, z0 = new_origin
    affine_gsvr = torch.tensor(
        [[sx, 0, 0, x0], [0, sy, 0, y0], [0, 0, sz, z0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    Dg, Hg, Wg = shape_rec
    z = torch.arange(Dg, device=device, dtype=torch.float32)
    y = torch.arange(Hg, device=device, dtype=torch.float32)
    x = torch.arange(Wg, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    vox_gsvr = torch.stack([xx, yy, zz, torch.ones_like(xx)], dim=-1).reshape(-1, 4).T
    affine_svort = torch.tensor(svort_affine, dtype=torch.float32, device=device)
    world = affine_gsvr @ vox_gsvr
    affine_inv = torch.linalg.inv(affine_svort.detach().cpu()).to(affine_svort.device)
    vox_svort = affine_inv @ world
    xs = vox_svort[0]
    ys = vox_svort[1]
    zs = vox_svort[2]
    D, H, W = svort_volume_np.shape
    xs = 2 * (xs / (W - 1)) - 1
    ys = 2 * (ys / (H - 1)) - 1
    zs = 2 * (zs / (D - 1)) - 1
    grid = torch.stack([xs, ys, zs], dim=-1)
    grid = grid.reshape(Dg, Hg, Wg, 3)
    svort_t = torch.from_numpy(svort_volume_np).to(dtype=torch.float32, device=device)[
        None, None
    ]
    svort_resampled = F.grid_sample(
        svort_t, grid[None], mode="bilinear", padding_mode="zeros", align_corners=True
    )
    svort_vol_gsvr = svort_resampled[0, 0]
    return (svort_vol_gsvr, affine_gsvr)


def run_svort_step(svort_cfg, subject_dict):
    """Infer slice poses and the prior using the G3R pretrained model."""
    checkpoint = Path(svort_cfg["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parent / checkpoint
    args = Namespace(
        use_svort=True,
        model_variant=svort_cfg["model_variant"],
        checkpoint_path=str(checkpoint),
        resolution_volume=svort_cfg["resolution_volume"],
        device=svort_cfg["device"],
    )
    subject_obj = Subject(
        name=subject_dict["name"],
        input_stacks=subject_dict["input_stacks"],
        input_masks=subject_dict.get("input_masks", []),
        device=f"cuda:{args.device}",
    )
    stacks = subject_obj.input_stacks
    if not stacks:
        raise ValueError("At least one input stack is required.")
    _, volume, stacks_out = _register(args, stacks)
    assert volume.dim() == 5
    volume = volume.squeeze(0).squeeze(0)
    volume_np = volume.detach().cpu().numpy()
    depth, height, width = volume_np.shape
    resolution = args.resolution_volume
    affine = np.diag([resolution, resolution, resolution, 1.0]).astype(np.float32)
    affine[:3, 3] = -np.array([width - 1, height - 1, depth - 1]) * resolution / 2
    if svort_cfg.get("save_volume", False):
        save_dir = Path(svort_cfg.get("save_dir", "outputs/prior"))
        if not save_dir.is_absolute():
            save_dir = Path(__file__).resolve().parent / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(volume_np.transpose(2, 1, 0), affine),
            save_dir / f"{subject_dict['name']}_svort.nii.gz",
        )
    return volume_np, affine, stacks_out


def _build_fg_index(vol, thresh=0.05):
    v = vol.float()
    m = v > thresh * v.max()
    return torch.nonzero(m, as_tuple=False)


def pretrain_on_volume_new(
    gs,
    target_vol,
    affine,
    optimizer,
    device,
    num_epochs=200,
    samples_per_iter=8192,
    k=32,
    log_scale_target=-2.0,
    reg_weight=0.01,
):
    gs.train()
    target_vol = target_vol.to(device)
    affine = affine.to(device)
    if torch.isnan(target_vol).any():
        print("[Pretrain Error] Target volume contains NaN!")
        return
    fg_index = _build_fg_index(target_vol, thresh=0.01)
    if len(fg_index) == 0:
        print("[Pretrain Error] No foreground voxels found.")
        return
    print(
        f"[Pretrain] Start pretraining on {len(fg_index)} voxels. Target scale: {log_scale_target:.3f}"
    )
    for ep in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        rand_idx = torch.randint(0, fg_index.shape[0], (samples_per_iter,))
        sel = fg_index[rand_idx]
        z, y, x = (sel[:, 0], sel[:, 1], sel[:, 2])
        gt = target_vol[z, y, x]
        vox = torch.stack(
            [x.float() + 0.5, y.float() + 0.5, z.float() + 0.5, torch.ones_like(x)],
            dim=1,
        )
        world = (affine @ vox.t()).t()[:, :3]
        topk = simple_knn_torch(world, gs.mu, k=k)
        pred, _, _ = gs(world, top_k_idcs=topk)
        if pred.shape != gt.shape:
            pred = pred.view_as(gt)
        loss_mse = F.mse_loss(pred, gt)
        loss_reg = reg_weight * ((gs.scaling - log_scale_target) ** 2).mean()
        loss = loss_mse + loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gs.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            gs.color.data.clamp_(min=0.0)
        if ep % 50 == 0:
            print(
                f"[Pretrain] Ep {ep:03d} | MSE: {loss_mse.item():.5f} | Reg: {loss_reg.item():.5f}"
            )


def load_data_from_svort_stacks(stacks_out, psf_scale_fac=1.0):
    """Use each registered slice pose directly, including its rotated PSF."""
    all_values = []
    all_coords_wrld = []
    all_slice_centers_wrld = []
    all_slice_idcs = []
    all_sigma_psf = []
    slice_id = 0
    for stack_idx, stack in enumerate(stacks_out):
        for local_idx in range(len(stack)):
            slice_obj = stack[local_idx]
            if not bool(slice_obj.mask.any()):
                continue
            coords = slice_obj.xyz_masked.detach().cpu().numpy().astype(np.float32)
            values = slice_obj.v_masked.detach().cpu().numpy().astype(np.float32)
            pose = (
                slice_obj.transformation.matrix(trans_first=False)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            rotation = pose[:, :3]
            center = pose[:, 3]
            spacing = np.array(
                [
                    float(slice_obj.resolution_x),
                    float(slice_obj.resolution_y),
                    float(slice_obj.resolution_z),
                ],
                dtype=np.float32,
            )
            sigma = generate_cov_psf(
                rotation,
                spacing,
                slice_thickness=float(slice_obj.resolution_z),
                scale_factor=psf_scale_fac,
            )
            all_coords_wrld.append(coords)
            all_values.append(values)
            all_slice_centers_wrld.append(
                np.broadcast_to(center, (len(coords), 3)).copy()
            )
            all_slice_idcs.append(np.full(len(coords), slice_id, dtype=np.int32))
            all_sigma_psf.append(np.broadcast_to(sigma, (len(coords), 3, 3)).copy())
            slice_id += 1
    coords_wrld = np.concatenate(all_coords_wrld, axis=0)
    values = np.concatenate(all_values, axis=0)
    slice_centers_wrld = np.concatenate(all_slice_centers_wrld, axis=0)
    slice_idcs = np.concatenate(all_slice_idcs, axis=0)
    sigma_psf = np.concatenate(all_sigma_psf, axis=0)
    percentile_99 = float(np.percentile(values, 99.9))
    values = np.clip(values, 0.0, percentile_99)
    max_value = float(values.max())
    values = values / max_value
    xyz_min = coords_wrld.min(axis=0)
    xyz_max = coords_wrld.max(axis=0)
    frame_center = (xyz_min + xyz_max) / 2.0
    coords_wrld = coords_wrld - frame_center
    slice_centers_wrld = slice_centers_wrld - frame_center
    xyz_min = coords_wrld.min(axis=0)
    xyz_max = coords_wrld.max(axis=0)
    bbox_wrld = (xyz_max - xyz_min) * 1.1
    new_origin = -bbox_wrld / 2.0
    print(
        f"[Slice poses] Loaded {slice_id} registered slices and {len(values)} foreground samples without NIfTI conversion."
    )
    print(f"[Slice poses] Removed frame centre: {frame_center}")
    print(f"[Slice poses] Coordinate min/max: {xyz_min} / {xyz_max}")
    return (
        None,
        [],
        [],
        slice_idcs,
        slice_centers_wrld,
        coords_wrld,
        values,
        sigma_psf,
        bbox_wrld,
        new_origin,
        0.0,
        max_value,
        frame_center,
    )


def g3r(
    stack_paths=None, mask_paths=None, config=None, output_file_path=None, subject=None
):
    # Preserve the original reconstruction seed and optimization schedules.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    svort_cfg = config.get("svort", {})
    use_svort = svort_cfg.get("enable", False)
    enable_prior = config["svort"].get("use_prior", False)
    distill_cfg = config.get("training", {}).get("svort_distillation", {})
    is_single_stack = len(stack_paths) == 1
    if is_single_stack:
        distill_enabled = bool(distill_cfg.get("single_stack_enabled", True))
    else:
        distill_enabled = bool(
            distill_cfg.get("multi_stack_enabled", distill_cfg.get("enabled", False))
        )
    distill_single_stack_lambda = float(distill_cfg.get("single_stack_lambda", 5.0))
    distill_lambda_start = float(distill_cfg.get("lambda_start", 0.5))
    distill_lambda_end = float(distill_cfg.get("lambda_end", 0.05))
    distill_warmup_end = int(distill_cfg.get("warmup_end", 300))
    distill_decay_end = int(distill_cfg.get("decay_end", 500))
    distill_samples = int(distill_cfg.get("samples", 4096))
    if (
        not is_single_stack
        and distill_enabled
        and (distill_decay_end <= distill_warmup_end)
    ):
        raise ValueError("svort_distillation.decay_end must be greater than warmup_end")
    lrs = config["training"]["learning_rates"]
    device = torch.device(f"cuda:{svort_cfg['device']}")
    max_epochs = config["training"]["max_epochs"] + 1
    num_gaussians = int(config["model"]["num_gaussians"])
    print("Num of Gaussians:", num_gaussians)
    D = 3
    M_NEIGHBORS = config["training"]["neighbors"]
    TOP_K_EVERY = config["training"]["top_k_every"]
    PSF = config["flags"]["use_psf"]
    MC = config["flags"]["use_motion_correction"]
    PSF_SCALE_FAC = config["reconstruction"]["psf_scale_factor"]
    SLICE_SCALING = config["flags"]["use_slice_scaling"]
    SLICE_UNCERTAINTY = config["flags"]["use_slice_uncertainty"]
    spacing_rec = config["reconstruction"]["spacing"]
    l2_lambda = config["training"]["loss_weights"]["lambda_l2_scale"]
    print(f"[Check] L2 Scale Regularization Lambda: {l2_lambda:.4f}")
    log_scale_target = config["training"]["loss_weights"]["log_scale_target"]
    svort_volume_np = None
    svort_affine = None
    svort_stacks_out = None
    if not use_svort or not enable_prior:
        raise ValueError(
            "This runtime requires svort.enable and svort.use_prior to be true."
        )
    if subject is None:
        raise ValueError("A subject with input_stacks is required.")
    print("[G3R] Running the pretrained model...")
    svort_volume_np, svort_affine, svort_stacks_out = run_svort_step(svort_cfg, subject)
    print("[Slice poses] Using the registered rigid pose of every slice.")
    (
        gt_nii,
        imgs,
        affines,
        slice_ids,
        slice_centers_wrld_np,
        coords_wrld_np,
        values_np,
        Sigma_psf,
        bbox_wrld,
        new_origin,
        min_value,
        max_value,
        direct_pose_frame_center,
    ) = load_data_from_svort_stacks(svort_stacks_out, psf_scale_fac=PSF_SCALE_FAC)
    svort_affine = svort_affine.copy()
    # Move the volume prior and slice samples into the same centred frame.
    svort_affine[:3, 3] -= direct_pose_frame_center
    print(f"[Slice poses] Reconstruction origin: {new_origin}")
    print(f"[Slice poses] Bounding box extent: {bbox_wrld}")
    shape_rec = [math.ceil(b / s) for b, s in zip(bbox_wrld, spacing_rec)]
    bbox_wrld = np.array([d * s for d, s in zip(shape_rec, spacing_rec)])
    if enable_prior and use_svort and (svort_volume_np is not None):
        print("[Resampling] Resampling the pretrained model prior to Gaussian space...")
        svort_affine_aligned = svort_affine.copy()
        print(f"[Slice poses] Reconstruction origin: {new_origin}")
        svort_vol_gsvr, svort_affine_gsvr = resample_svort_to_gsvr_space(
            svort_volume_np,
            svort_affine_aligned,
            shape_rec=shape_rec,
            spacing_rec=spacing_rec,
            new_origin=new_origin,
            device=device,
        )
        print(f"[Resampling] Done.")
        s_data = svort_vol_gsvr
        t_mean = values_np.mean()
        t_std = values_np.std()
        mask_s = s_data > 0.001
        if mask_s.sum() > 0:
            s_mean = s_data[mask_s].mean()
            s_std = s_data[mask_s].std()
            print(
                f"[Intensity] Scaling the pretrained model prior: ({s_mean:.2f} +/- {s_std:.2f}) -> ({t_mean:.2f} +/- {t_std:.2f})"
            )
            s_data = (s_data - s_mean) / (s_std + 1e-08) * t_std + t_mean
            s_data = torch.clamp(s_data, min=0.0, max=max_value)
            s_data = torch.where(mask_s, s_data, torch.zeros_like(s_data))
            svort_vol_gsvr = s_data
        svort_vol_gsvr_np = svort_vol_gsvr.cpu().numpy()
        svort_affine_gsvr_np = svort_affine_gsvr.cpu().numpy()
    vis_grid_flat_3D = create_vis_grid(shape_rec, bbox_wrld, device)
    print(f"Number of non-zero voxels in the raw data: {len(values_np)}")
    gs = GSVR(
        num_gaussians,
        np.max(slice_ids) + 1,
        D=D,
        bbox_wrld=torch.from_numpy(bbox_wrld).to(dtype=torch.float32, device=device),
        mc=MC,
        apply_slice_scaling=SLICE_SCALING,
        apply_slice_uncertainty=SLICE_UNCERTAINTY,
    )
    print("Using G3R")
    gs.initialize_parameters_from_svortvolume_new(
        svort_vol_gsvr_np, svort_affine_gsvr_np, device=device
    )
    print("mu mean/std:", gs.mu.mean().item(), gs.mu.std().item())
    print("color mean/std:", gs.color.mean().item(), gs.color.std().item())
    gs.to(device)
    print(f"Reconstructing on device: {device}")
    optimizer, scheduler, grad_scaler = _init_optim(
        gs, MC, SLICE_SCALING, SLICE_UNCERTAINTY, max_epochs, lrs
    )
    loss_fn = nn.L1Loss()
    ressource = faiss.StandardGpuResources()
    faiss_config = faiss.GpuIndexFlatConfig()
    faiss_config.device = device.index
    gpu_index = faiss.GpuIndexFlatL2(ressource, D, faiss_config)
    coords_wrld = torch.from_numpy(coords_wrld_np).float().to(device)
    slice_centers_wrld = (
        torch.from_numpy(slice_centers_wrld_np).float().to(device) if MC else None
    )
    values = torch.from_numpy(values_np).float().to(device)
    Sigma_psf = torch.from_numpy(Sigma_psf).float().to(device) if PSF else None
    slice_ids = torch.from_numpy(slice_ids).int().to(device).contiguous()
    if enable_prior and svort_volume_np is not None:
        pretrain_on_volume_new(
            gs=gs,
            target_vol=svort_vol_gsvr,
            affine=svort_affine_gsvr,
            optimizer=optimizer,
            device=device,
            num_epochs=201,
            samples_per_iter=8192,
            k=M_NEIGHBORS,
            log_scale_target=log_scale_target,
            reg_weight=0.01,
        )
        gs.scaling.requires_grad = True
        gs.color.requires_grad = True
        optimizer, scheduler, grad_scaler = _init_optim(
            gs, MC, SLICE_SCALING, SLICE_UNCERTAINTY, max_epochs, lrs
        )

        gpu_index.reset()
        gpu_index.add(gs.mu.data)
    print("Compiling Gaussian reconstruction model...")
    gs_compiled = torch.compile(gs, mode="reduce-overhead")
    if MC:
        print("[Fit] Resetting motion parameters to identity...")
        with torch.no_grad():
            gs.translation_mc.data.fill_(0.0)
            gs.rotation_mc.data[:, 0] = 1.0
            gs.rotation_mc.data[:, 1:] = 0.0
    distill_active = (
        distill_enabled and enable_prior and use_svort and (svort_volume_np is not None)
    )
    if distill_active and is_single_stack:
        print(
            f"[Distill] Single-stack pretrained model prior constraint enabled: fixed lambda={distill_single_stack_lambda}."
        )
    elif distill_active:
        print(
            f"[Distill] Multi-stack pretrained model prior constraint enabled: lambda={distill_lambda_start} until epoch {distill_warmup_end}, decay to {distill_lambda_end} at epoch {distill_decay_end}."
        )
    elif not distill_enabled:
        stack_mode = "single-stack" if is_single_stack else "multi-stack"
        print(f"[Distill] Disabled for {stack_mode} input by configuration.")
    else:
        print(
            "[Distill] Requested, but the pretrained model prior is unavailable; distillation disabled."
        )
    t0 = time.time()
    total_time = 0
    for i in range(max_epochs):
        batch_coords_wrld = coords_wrld.contiguous()
        batch_slice_centers_wrld = slice_centers_wrld
        batch_values = values
        batch_Sigma_psf = Sigma_psf
        batch_slice_ids = slice_ids
        loss_history = {"sr": [], "sr_reg": [], "mc": []}
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            if (
                MC
                and batch_slice_centers_wrld is not None
                and (batch_slice_ids is not None)
            ):
                batch_coords_wrld, batch_Sigma_psf_tf, batch_q, batch_t = (
                    gs.motion_correction_fused(
                        batch_coords_wrld,
                        batch_slice_ids,
                        batch_slice_centers_wrld,
                        Sigma_psf=batch_Sigma_psf,
                    )
                )
                batch_Sigma_psf = (
                    batch_Sigma_psf_tf
                    if batch_Sigma_psf_tf is not None
                    else batch_Sigma_psf
                )
            else:
                batch_q = torch.tensor([0.0])
                batch_t = torch.tensor([0.0])
            if i % TOP_K_EVERY == 0:
                gpu_index.reset()
                gpu_index.add(gs.mu.data)
                _, top_k_idcs = gpu_index.search(batch_coords_wrld, M_NEIGHBORS)
                batch_top_k_idcs = top_k_idcs
                if (top_k_idcs >= gs.num_gaussians).any() or (top_k_idcs < 0).any():
                    print("Error: Faiss returned invalid indices!")
                    if torch.isnan(gs.mu).any():
                        print(
                            "!!! DETECTED NaN in Gaussian Means! Training might be unstable."
                        )
                    top_k_idcs = torch.clamp(
                        top_k_idcs, min=0, max=gs.num_gaussians - 1
                    )
            color_pred, _, _ = gs_compiled(
                batch_coords_wrld,
                slice_ids=batch_slice_ids,
                slice_centers=batch_slice_centers_wrld,
                Sigma_psf=batch_Sigma_psf,
                top_k_idcs=batch_top_k_idcs,
            )
            if SLICE_SCALING:
                slice_scales = gs_compiled.slice_scaling
                slice_scales_softplus = torch.nn.functional.softplus(slice_scales)
                slice_scales_softplus = (
                    slice_scales_softplus / slice_scales_softplus.mean()
                )
                color_pred = color_pred * slice_scales_softplus[batch_slice_ids]
            if SLICE_UNCERTAINTY:
                slice_weights = gs_compiled.slice_weight[batch_slice_ids]
                slice_weights = torch.nn.functional.softplus(slice_weights)
                slice_weights = slice_weights.clamp(min=0.001)
                sw_reg = 0.1 * -torch.log(slice_weights + 1e-08).mean()
                l1_diff = torch.abs(color_pred - batch_values)
                loss_sr = (l1_diff * slice_weights).mean() + sw_reg
                l1_diff = l1_diff.mean()
            else:
                loss_sr = loss_fn(color_pred, batch_values)
                l1_diff = loss_sr
            loss_distill = torch.tensor(0.0, device=device)
            lambda_distill = 0.0
            if distill_active:
                rand_coords = torch.rand(distill_samples, 3, device=device)
                bbox_tensor = torch.tensor(bbox_wrld, device=device).float()
                origin_tensor = torch.tensor(new_origin, device=device).float()
                distill_coords_world = rand_coords * bbox_tensor + origin_tensor
                inv_affine = torch.linalg.inv(svort_affine_gsvr)
                distill_coords_homo = torch.cat(
                    [
                        distill_coords_world,
                        torch.ones(distill_samples, 1, device=device),
                    ],
                    dim=1,
                )
                distill_voxels = (inv_affine @ distill_coords_homo.T).T[:, :3]
                vol_d, vol_h, vol_w = svort_vol_gsvr.shape
                norm_z = 2 * (distill_voxels[:, 2] / (vol_d - 1)) - 1
                norm_y = 2 * (distill_voxels[:, 1] / (vol_h - 1)) - 1
                norm_x = 2 * (distill_voxels[:, 0] / (vol_w - 1)) - 1
                grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).view(
                    1, 1, 1, distill_samples, 3
                )
                svort_target_values = F.grid_sample(
                    svort_vol_gsvr.view(1, 1, vol_d, vol_h, vol_w),
                    grid,
                    align_corners=True,
                ).view(-1)
                _, distill_top_k = gpu_index.search(distill_coords_world, M_NEIGHBORS)
                distill_top_k = torch.clamp(
                    distill_top_k, min=0, max=gs.num_gaussians - 1
                )
                distill_pred, _, _ = gs(
                    distill_coords_world,
                    slice_ids=None,
                    top_k_idcs=distill_top_k,
                    scale_scale=1.0,
                )
                mask_valid = svort_target_values > 0.01
                if mask_valid.sum() > 0:
                    loss_distill = F.l1_loss(
                        distill_pred[mask_valid], svort_target_values[mask_valid]
                    )
                if is_single_stack:
                    lambda_distill = distill_single_stack_lambda
                elif i < distill_warmup_end:
                    lambda_distill = distill_lambda_start
                elif i < distill_decay_end:
                    progress = (i - distill_warmup_end) / (
                        distill_decay_end - distill_warmup_end
                    )
                    lambda_distill = (
                        distill_lambda_start * (1.0 - progress)
                        + distill_lambda_end * progress
                    )
                else:
                    lambda_distill = distill_lambda_end
            loss_mc = (batch_q**2).mean() + (batch_t**2).mean()
            lambda_mc = 0.01
            loss_sr_reg = l2_lambda * ((gs.scaling - log_scale_target) ** 2).mean()
            loss = loss_sr + loss_sr_reg + lambda_distill * loss_distill
        loss_history["sr"].append(l1_diff.item())
        loss_history["sr_reg"].append(loss_sr_reg.item())
        loss_history["mc"].append(loss_mc.item())
        loss_history["distill"] = loss_distill.item()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        gs.color.data.clamp_(min=0.0, max=None)
        loss_mean = {
            "sr": np.array(loss_history["sr"]).mean(),
            "sr_reg": np.array(loss_history["sr_reg"]).mean(),
            "mc": np.array(loss_history["mc"]).mean(),
            "distill": np.array(loss_history["distill"]).mean(),
        }
        if scheduler is not None:
            scheduler.step()
        if i == 0:
            t0 = time.time()
        if i % 50 == 0 or i == max_epochs - 1:
            t1 = time.time()
            t_delta = t1 - t0
            total_time += t_delta
            print(
                f"Epoch {i} loss sr: {loss_mean['sr']:.6f}, loss sr_reg: {loss_mean['sr_reg']:.6f}, loss mc: {loss_mean['mc']:.6f}, loss ds: {loss_mean['distill']:.6f}, lambda:{lambda_distill:.6f}, time per epoch: {t_delta / 50:.2f}s, total training time: {total_time:.2f}s"
            )
            t0 = time.time()
        if i == max_epochs - 1:
            print(f"Visualizing epoch {i}...")
            visualize_gaussians(
                gt_nii,
                gs,
                gpu_index,
                vis_grid_flat_3D,
                shape_rec,
                spacing_rec,
                new_origin,
                epoch=i,
                k=M_NEIGHBORS,
                psf_scale_fac=1.0,
                min_value=min_value,
                max_value=max_value,
                output_file_path=output_file_path,
                PSF=PSF,
                ttime=total_time,
                scale_scale=1.0,
                scale_threshold=0.01,
            )
            t0 = time.time()


def generate_cov_psf(affine3x3, spacing, slice_thickness=None, scale_factor=1.0):
    fwhm_to_std = 1.2 / (2 * np.sqrt(2 * np.log(2))) * scale_factor
    sigma_psf_diag = np.array(spacing) * fwhm_to_std
    if slice_thickness is not None:
        sigma_psf_diag[2] = slice_thickness * fwhm_to_std / 1.2
    Sigma_psf = np.diag(sigma_psf_diag**2)
    L = affine3x3[:3, :3] / np.linalg.norm(affine3x3[:3, :3], axis=0)
    Sigma_psf = L @ Sigma_psf @ L.T
    return Sigma_psf
