"""G3R pretrained priors and their original architecture dependencies."""

from typing import Optional, Dict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..svr.reconstruction import cg
from ..transform import RigidTransform, mat_update_resolution, point2mat, mat2point, axisangle2mat
from .attention import TransformerEncoder, PositionalEncoding, ResNet
from ..slice_acquisition import slice_acquisition, slice_acquisition_adjoint


class SVRtransformerV2(nn.Module):
    def __init__(
        self,
        n_res=50,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=6,
        d_model=256,
        d_inner=512,
        dropout=0.1,
        n_channels=2,
    ):
        super().__init__()
        # self.img_encoder = ViT(channels=n_channels, num_classes=d_model)
        self.img_encoder = ResNet(
            n_res=n_res, d_model=d_model, pretrained=False, d_in=n_channels + 2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.fc_score = nn.Linear(d_model, 1)

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            slices = torch.cat(
                [
                    slices,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        return slices

    def forward(self, theta, slices, pos, volume, params, attn_mask=None):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    False,
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        if isinstance(self.img_encoder, ResNet):
            slices = self.pos_augment(slices, y)
        else:  # ViT
            if y is not None:
                slices = torch.cat([slices, y], 1)
        x = self.img_encoder(slices)
        x, attn = self.encoder(x, pe, attn_mask)
        dtheta = self.fc(x)

        score = self.fc_score(x)
        score = F.softmax(score, dim=0) * score.shape[0]
        score = torch.clamp(score, max=3.0)

        return theta + dtheta, score, attn

class SRR(nn.Module):
    def __init__(
        self,
        n_iter: int = 10,
        tol: float = 0.0,
        output_relu: bool = True,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.output_relu = output_relu

    def forward(
        self,
        transforms: torch.Tensor,
        slices: torch.Tensor,
        volume: torch.Tensor,
        params: Dict,
        p: Optional[torch.Tensor] = None,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        At = lambda x: self.At(transforms, x, params, slices_mask, vol_mask)
        AtA = lambda x: self.AtA(transforms, x, p, params, slices_mask, vol_mask)

        b = At(slices * p if p is not None else slices)
        volume = cg(AtA, b, volume, self.n_iter, self.tol)

        if self.output_relu:
            return F.relu(volume, True)
        else:
            return volume

    def A(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def At(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def AtA(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        p: Optional[torch.Tensor],
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        slices = self.A(transforms, x, params, slices_mask, vol_mask)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, params, slices_mask, vol_mask)
        return vol

def dot(x, y):
    return torch.dot(x.flatten(), y.flatten())

def CG(A, b, x0, n_iter):
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new

class SRR_svort(nn.Module):
    def __init__(self, n_iter=10, use_CG=False, alpha=0.5, beta=0.02, delta=0.1):
        super().__init__()
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta * delta * delta
        self.delta = delta
        self.use_CG = use_CG

    def forward(self, theta, slices, volume, params, p=None, mu=0, z=None, vol_mask=None, slices_mask=None):
        if len(theta.shape) == 2:
            transforms = axisangle2mat(theta)
        else:
            transforms = theta

        A = lambda x:self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x:self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x:self.AtA(transforms, x, vol_mask, slices_mask, p, params, mu, z)

        x = volume
        y = slices

        if self.use_CG:
            b = At(y * p if p is not None else y)
            if mu and z is not None:
                b = b + mu*z
            x = CG(AtA, b, volume, self.n_iter)
        else:
            for _ in range(self.n_iter):
                err = A(x) - y
                if p is not None:
                    err = p * err
                g = At(err)
                if self.beta:
                    dR = self.dR(x, self.delta)
                    g.add_(dR, alpha=self.beta)
                x.add_(g, alpha=-self.alpha)
        return F.relu(x, True)

    def A(self, transforms, x, vol_mask, slices_mask, params):
        return slice_acquisition(transforms, x, vol_mask, slices_mask, params['psf'], params['slice_shape'], params['res_s'] / params['res_r'], False, params['interp_psf'])

    def At(self, transforms, x, slices_mask, vol_mask, params):
        return slice_acquisition_adjoint(transforms, params['psf'], x, slices_mask, vol_mask, params['volume_shape'], params['res_s'] / params['res_r'], params['interp_psf'], False)

    def AtA(self, transforms, x, vol_mask, slices_mask, p, params, mu, z):
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if mu and z is not None:
            vol = vol + mu * x
        return vol

    def dR(self, v, delta):
        g = torch.zeros_like(v)
        D, H, W = v.shape[-3:]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    v0 = v[:, :, 1:D-1, 1:H-1, 1:W-1]
                    v1 = v[:, :, 1+dz:D-1+dz, 1+dy:H-1+dy, 1+dx:W-1+dx]
                    dv = v0 - v1
                    dv_ = dv * (1 / (dx*dx + dy*dy + dz*dz) / (delta*delta))
                    g[:, :, 1:D-1, 1:H-1, 1:W-1] += dv_ / torch.sqrt(1 + dv * dv_)
        return g

def PSFreconstruction(transforms, slices, slices_mask, vol_mask, params, equalize=True):
    return slice_acquisition_adjoint(transforms, params['psf'], slices, slices_mask, vol_mask, params['volume_shape'], params['res_s'] / params['res_r'], params['interp_psf'], equalize)

class EZ_ResUNetRefiner(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        print(f"G3R prior v1 refiner initialized (base channels: {base_channels})")

        # --- Level 1 (Full Res) ---
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1 = ResBlock3D_UNet(base_channels)

        # --- Level 2 (Bottleneck, 1/2 Res) ---
        # Down 1
        self.down1 = Downsample(base_channels, base_channels * 2)
        self.enc2 = ResBlock3D_UNet(base_channels * 2)

        # Down 2 -> Bottleneck (1/4 Res)
        self.down2 = Downsample(base_channels * 2, base_channels * 4)

        # Bottleneck: 既然层数少，这里多堆两个 Block 增强非线性能力
        self.bottleneck = nn.Sequential(
            ResBlock3D_UNet(base_channels * 4),
            ResBlock3D_UNet(base_channels * 4),
            ResBlock3D_UNet(base_channels * 4)
        )

        # --- Decoder ---

        # Up 2 -> Level 2
        self.up2 = Upsample(base_channels * 4, base_channels * 2)
        self.dec2 = ResBlock3D_UNet(base_channels * 2)

        # Up 1 -> Level 1
        self.up1 = Upsample(base_channels * 2, base_channels)
        self.dec1 = ResBlock3D_UNet(base_channels)

        # --- Output ---
        self.tail = nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1)

        # Zero Init
        nn.init.constant_(self.tail.weight, 0)
        nn.init.constant_(self.tail.bias, 0)

    def forward(self, volume):
        # --- Encoder ---
        x1 = self.conv_in(volume)
        x1 = self.enc1(x1)           # Full Res

        x2 = self.down1(x1)
        x2 = self.enc2(x2)           # 1/2 Res

        # --- Bottleneck ---
        x_bot = self.down2(x2)       # 1/4 Res (最低分辨率)
        x_bot = self.bottleneck(x_bot)

        # --- Decoder ---
        d2 = self.up2(x_bot)
        d2 = match_tensor_size(d2, x2)
        d2 = d2 + x2                 # Skip
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = match_tensor_size(d1, x1)
        d1 = d1 + x1                 # Skip
        d1 = self.dec1(d1)

        out = self.tail(d1)

        return volume + out

class EZ_ResUNetRefiner_V6(nn.Module):
    def __init__(self, in_channels=2, base_channels=32):
        super().__init__()
        print(f"G3R prior v2 refiner initialized (base channels: {base_channels})")

        # --- Level 1 (Full Res) ---
        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1 = ResBlock3D_UNet(base_channels)

        # --- Level 2 (Bottleneck, 1/2 Res) ---
        # Down 1
        self.down1 = Downsample(base_channels, base_channels * 2)
        self.enc2 = ResBlock3D_UNet(base_channels * 2)

        # Down 2 -> Bottleneck (1/4 Res)
        self.down2 = Downsample(base_channels * 2, base_channels * 4)

        # Bottleneck: 既然层数少，这里多堆两个 Block 增强非线性能力
        self.bottleneck = nn.Sequential(
            ResBlock3D_UNet(base_channels * 4),
            ResBlock3D_UNet(base_channels * 4),
            ResBlock3D_UNet(base_channels * 4)
        )

        # --- Decoder ---

        # Up 2 -> Level 2
        self.up2 = Upsample(base_channels * 4, base_channels * 2)
        self.dec2 = ResBlock3D_UNet(base_channels * 2)

        # Up 1 -> Level 1
        self.up1 = Upsample(base_channels * 2, base_channels)
        self.dec1 = ResBlock3D_UNet(base_channels)

        # --- Output ---
        # 把双通道压缩为单通道
        self.tail = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1)

        # Zero Init
        nn.init.constant_(self.tail.weight, 0)
        nn.init.constant_(self.tail.bias, 0)
        self.tanh = nn.Tanh()
    def forward(self, volume, mask_vol=None):
        # --- Encoder ---
        x_in = torch.cat([volume, mask_vol], dim=1)

        x1 = self.conv_in(x_in)
        x1 = self.enc1(x1)           # Full Res

        x2 = self.down1(x1)
        x2 = self.enc2(x2)           # 1/2 Res

        # --- Bottleneck ---
        x_bot = self.down2(x2)       # 1/4 Res (最低分辨率)
        x_bot = self.bottleneck(x_bot)

        # --- Decoder ---
        d2 = self.up2(x_bot)
        d2 = match_tensor_size(d2, x2)
        d2 = d2 + x2                 # Skip
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = match_tensor_size(d1, x1)
        d1 = d1 + x1                 # Skip
        d1 = self.dec1(d1)

        raw_residual = self.tail(d1)

        # 计算置信度 (Confidence Map):1 - exp(-k * x)
        # 软门控 (Soft Gating)

        # mask=0.1 -> shifted=0.0 -> Conf=0.00 ( Refiner 重绘)
        # mask=0.4 -> shifted=0.3 -> Conf=0.78 (单层中心，保留纹理)
        # mask=1.3 -> shifted=1.2 -> Conf=0.99 (多层，完全信任)
        threshold = 0.1
        mask_shifted = torch.relu(mask_vol - threshold)
        k = 7
        conf = 1.0 - torch.exp(-k * mask_shifted)  # [0, 1]

        alpha = 0.02  # 信度高区域，保留原样
        beta = 1.0   # 信度低区域，允许修改

        # 线性插值
        gate_map = conf * alpha + (1.0 - conf) * beta

        out = volume + raw_residual * gate_map
        return out

def match_tensor_size(x_up, x_enc):
    """
    如果 x_up (上采样结果) 和 x_enc (Encoder特征) 尺寸不一致，
    使用插值将 x_up 调整为 x_enc 的尺寸。
    """
    if x_up.shape[2:] != x_enc.shape[2:]:
        # align_corners=False 对于 trilinear 是标准做法
        return F.interpolate(x_up, size=x_enc.shape[2:], mode='trilinear', align_corners=False)
    return x_up

class ResBlock3D_UNet(nn.Module):
    """
    Standard Pre-activation ResBlock for 3D
    GN -> ReLU -> Conv -> GN -> ReLU -> Conv
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),

            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return x + self.net(x)

class Downsample(nn.Module):
    """
    Downsampling using Strided Convolution (Learnable)
    Better than MaxPool for reconstruction tasks.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """
    Artifact-free Upsampling:
    Trilinear Interpolate -> Convolution
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Scale factor = 2
        # x_up = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x_up = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x_up)

class SVoRTv5(nn.Module):
    def __init__(self, n_iter=4, iqa=True, vol=True, pe=True):
        super().__init__()
        print("G3R prior v1 initialized")
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None
        self.n_iter = n_iter
        input_dim = 9 + 2  # no position encoding

        self.save_intermediate = False  # 开关
        self.save_dir = None            # 保存路径
        self.current_step = 0           # 当前迭代次

        self.svrnet1 = SVRtransformerV2(
            n_layers=4,
            n_head=4 * 2,
            d_in=input_dim,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=1,
        )

        self.svrnet2 = SVRtransformerV2(
            n_layers=4 * 2,
            n_head=4 * 2,
            d_in=input_dim,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=2,
        )

        if iqa:
            self.srr = SRR(n_iter=2) #, use_CG=True
            # self.refiner = ResUNetRefiner(in_channels=1, base_channels=32)    #UNetSRR    last iter use UNet
            self.refiner = EZ_ResUNetRefiner(in_channels=1, base_channels=32) #EZ ResUNet

    def set_debug_save(self, enable=False, save_dir=None, step=0):
        """
        在训练循环中调用此方法来控制是否保存中间结果
        """
        self.save_intermediate = enable
        self.save_dir = save_dir
        self.current_step = step

    def set_refiner_requires_grad(self, requires_grad=True):
        if hasattr(self, 'refiner'):
            print(f"[Model] Setting Refiner requires_grad = {requires_grad}")
            for param in self.refiner.parameters():
                param.requires_grad = requires_grad

            # 如果冻结，切换到 eval 模式 (影响 Dropout/BN/GN 等)
            if not requires_grad:
                self.refiner.eval()
            else:
                self.refiner.train()
        else:
            print("[Model] Warning: No refiner found to freeze/unfreeze.")

    def forward(self, data, n_iter=None):

        if n_iter is None:
            n_iter = self.n_iter

        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": False,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]
        attn_mask = None

        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0 + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None

        mask_stacks = None

        default_grad = torch.is_grad_enabled()
        if n_iter > 1:
            iter_grad = random.randint(1, n_iter - 1)
        else:
            iter_grad = 0

        for i in range(n_iter):
            if i != 0 and i != iter_grad:
                torch.set_grad_enabled(False)
            svrnet = self.svrnet2 if i else self.svrnet1
            theta, iqa_score, attn = svrnet(
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else volume.detach(),
                params,
                attn_mask,
            )

            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))

            with torch.no_grad():
                if (i == 0 or i == iter_grad) and "transforms_gt" in data:
                    # adjust for GT
                    diff = (
                        RigidTransform(data["transforms_gt"])
                        .compose(_trans.inv())
                        .axisangle()
                        .detach()
                    )
                    err_R = torch.sum(diff[:, :3] ** 2, -1)
                    err_T = torch.sum(diff[:, 3:] ** 2, -1)
                    err = 1000.0 * err_R + err_T
                    _, min_idx = torch.topk(err, 5, largest=False, sorted=False)
                    diff = diff[min_idx, :].mean(0, keepdim=True)
                else:
                    diff = None

            if diff is not None:
                _trans = RigidTransform(diff).compose(_trans)

            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = PSFreconstruction(mat, stacks, mask_stacks, None, params)

            if self.iqa:
                volume = self.srr(
                    mat, stacks, volume, params, iqa_score.view(-1, 1, 1, 1)
                )
                # 只在最后一次迭代配准使用CNN
                if i == n_iter - 1:
                    volume = self.refiner(volume)
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)
            torch.set_grad_enabled(default_grad)

        self.attn = attn.detach()

        return trans, volumes, thetas

class SVoRTv6(nn.Module):
    def __init__(self, n_iter=4, iqa=True, vol=True, pe=True):
        super().__init__()
        print("G3R prior v2 initialized")
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None
        self.n_iter = n_iter
        input_dim = 9 + 2  # no position encoding

        self.save_intermediate = False  # 开关
        self.save_dir = None            # 保存路径
        self.current_step = 0           # 当前迭代次

        self.svrnet1 = SVRtransformerV2(
            n_layers=4,
            n_head=4 * 2,
            d_in=input_dim,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=1,
        )

        self.svrnet2 = SVRtransformerV2(
            n_layers=4 * 2,
            n_head=4 * 2,
            d_in=input_dim,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=2,
        )

        if iqa:
            self.srr = SRR_svort(n_iter=2, use_CG=True)
            self.refiner = EZ_ResUNetRefiner_V6(in_channels=2, base_channels=32) #EZ ResUNet

    def set_debug_save(self, enable=False, save_dir=None, step=0):
        """
        在训练循环中调用此方法来控制是否保存中间结果
        """
        self.save_intermediate = enable
        self.save_dir = save_dir
        self.current_step = step

    def set_refiner_requires_grad(self, requires_grad=True):
        if hasattr(self, 'refiner'):
            print(f"[Model] Setting Refiner requires_grad = {requires_grad}")
            for param in self.refiner.parameters():
                param.requires_grad = requires_grad

            # 如果冻结，切换到 eval 模式 (影响 Dropout/BN/GN 等)
            if not requires_grad:
                self.refiner.eval()
            else:
                self.refiner.train()
        else:
            print("[Model] Warning: No refiner found to freeze/unfreeze.")

    def forward(self, data, n_iter=None):

        if n_iter is None:
            n_iter = self.n_iter

        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": False,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]
        attn_mask = None

        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0 + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None

        mask_stacks = None

        default_grad = torch.is_grad_enabled()
        if n_iter > 1:
            iter_grad = random.randint(1, n_iter - 1)
        else:
            iter_grad = 0

        for i in range(n_iter):
            if i != 0 and i != iter_grad:
                torch.set_grad_enabled(False)
            svrnet = self.svrnet2 if i else self.svrnet1
            theta, iqa_score, attn = svrnet(
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else volume.detach(),
                params,
                attn_mask,
            )

            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))

            with torch.no_grad():
                if (i == 0 or i == iter_grad) and "transforms_gt" in data:
                    # adjust for GT
                    diff = (
                        RigidTransform(data["transforms_gt"])
                        .compose(_trans.inv())
                        .axisangle()
                        .detach()
                    )
                    err_R = torch.sum(diff[:, :3] ** 2, -1)
                    err_T = torch.sum(diff[:, 3:] ** 2, -1)
                    err = 1000.0 * err_R + err_T
                    _, min_idx = torch.topk(err, 5, largest=False, sorted=False)
                    diff = diff[min_idx, :].mean(0, keepdim=True)
                else:
                    diff = None

            if diff is not None:
                _trans = RigidTransform(diff).compose(_trans)

            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = PSFreconstruction(mat, stacks, mask_stacks, None, params)

                # Mask：计算像素点是否被切片覆盖
                mask_vol = None
                if i == n_iter - 1: # 只在喂给 Refiner 前计算
                    ones_stacks = torch.ones_like(stacks)
                    mask_vol = PSFreconstruction(mat, ones_stacks, mask_stacks, None, params, equalize=False)
                    mask_vol = mask_vol.detach()

            if self.iqa:
                volume = self.srr(
                    mat, stacks, volume, params, iqa_score.view(-1, 1, 1, 1)
                )
                # 只在最后一次迭代配准使用CNN
                if i == n_iter - 1:
                    volume = self.refiner(volume,mask_vol)
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)
            torch.set_grad_enabled(default_grad)

        self.attn = attn.detach()

        return trans, volumes, thetas
