# G3R: Gaussian-Based Geometry-Guided Reconstruction for Fetal Brain MRI

**MICCAI 2026** · Zhibao Cai, Yao Lv, Xin Zhang, and Chaoxiang Yang

[Paper](https://papers.miccai.org/miccai-2026/0412-Paper2298.html) · [Pretrained models](https://github.com/Aeaitly/G3R/releases/tag/v1.0.0)

G3R reconstructs high-resolution fetal brain MRI from motion-corrupted thick slices using explicit 3D Gaussian primitives and a learned structural prior. The prior guides Gaussian initialization and subsequent slice-consistent optimization.

This release provides two pretrained structural priors, the reconstruction implementation, and a self-contained NeSVoR/SVoRT inference dependency. The reconstruction interface preserves individual slice poses when passing registration results to Gaussian reconstruction. Network training scripts are not included.

| Model | Refiner input | Refinement | Checkpoint |
| --- | --- | --- | --- |
| v1 | Coarse volume | Residual volume refinement | `g3r_v1.pt` |
| v2 (default) | Coarse volume and physical density map | Density-gated residual refinement | `g3r_v2.pt` |

The density-gated DGPN mechanism described in the paper corresponds to v2. Both models retain their original network definitions and checkpoint values; v1 provides an alternative prior without density gating.

## Installation

The tested software stack is Linux, Python 3.10, PyTorch 2.1.2 with CUDA 12.1, torchvision 0.16.2, NumPy 1.26.4, and GPU-enabled FAISS 1.13.2. Building the local CUDA extensions requires the CUDA toolkit (`nvcc`) and a compatible C++ compiler; our build uses CUDA toolkit 12.1 and GCC 11. A CUDA-enabled GPU is required.

```bash
git clone https://github.com/Aeaitly/G3R.git
cd G3R
conda create -n g3r python=3.10 numpy=1.26.4 pip -y
conda activate g3r
conda install -c pytorch -c nvidia faiss-gpu=1.13.2 -y
python -m pip install -r requirements.txt

cd third_party/nesvor
python setup.py build_ext --inplace
cd ../..
```

The extensions are built inside this checkout. The G3R entrypoint uses this local dependency without installing or replacing a global NeSVoR package. For an RTX 4090, `TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2` can be set before the build command to limit build targets and parallel compilation.

## Download the pretrained model

```bash
python download_weights.py
# To download both model variants:
python download_weights.py --model all
```

The default command downloads `pretrained/g3r_v2.pt`. Select `--model v1`, `--model v2`, or `--model all` to choose the downloads. Each file is verified against its SHA-256 digest and contains model parameters only.

## Reconstruct a volume

Copy `configs/g3r.yaml` for multiple stacks or `configs/g3r_single_stack.yaml` for a single stack. Replace the example input paths with your NIfTI stack and brain-mask paths. Use `input_masks: []` for inputs that are already brain-masked. Input intensities outside the brain should be zero when no masks are supplied. Slice thickness is read from each stack's NIfTI header.

```bash
python train_slice_pose.py --config configs/g3r.yaml
# To select v1 and its matching checkpoint:
python train_slice_pose.py --config configs/g3r.yaml --model v1
```

Despite the historical filename, this command performs **reconstruction** with a fixed pretrained network. It does not train the prior network. Registration failure stops reconstruction instead of silently falling back to another method.

The default model is v2. The `--model` option selects the network and its corresponding checkpoint together. Without that option, `svort.model_variant` and `svort.checkpoint_path` are read from the YAML configuration. The same selection applies to single-stack reconstruction using `configs/g3r_single_stack.yaml`.

Outputs are timestamped NIfTI volumes and a copy of the configuration under `outputs/<case>/`. Relative checkpoint and output paths are resolved from the repository root. `svort.device` selects the CUDA device visible to the process; for example, `CUDA_VISIBLE_DEVICES=1` with `svort.device: 0` uses physical GPU 1.

The example configurations use 80,000 Gaussians and a terminal epoch index of 500 (epochs 0 through 500). Prior distillation is **disabled for multiple stacks** and **enabled with a fixed weight for a single stack**. The annealing settings apply only if multi-stack distillation is explicitly enabled. These settings are editable in the YAML files. The internal prior-prefitting stage retains the original implementation's 201 iterations.

## Acknowledgments and license

The Gaussian reconstruction code builds on [GSVR](https://github.com/m-dannecker/Gaussian-Primitives-for-Fast-SVR) under the Apache License 2.0. The bundled image, transform, SVoRT, and slice-acquisition components build on [NeSVoR](https://github.com/daviddmc/NeSVoR) under the MIT License. See `LICENSE`, `NOTICE`, and `third_party/nesvor/LICENSE`.

## Citation

```bibtex
@inproceedings{cai2026g3r,
  author    = {Cai, Zhibao and Lv, Yao and Zhang, Xin and Yang, Chaoxiang},
  title     = {G3R: Gaussian-Based Geometry-Guided Reconstruction for Fetal Brain MRI},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year      = {2026},
  url       = {https://papers.miccai.org/miccai-2026/0412-Paper2298.html}
}
```
