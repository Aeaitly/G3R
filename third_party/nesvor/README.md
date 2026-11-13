# NeSVoR inference dependency for G3R

This directory vendors the NeSVoR utilities used by G3R inference.
The upstream project is [NeSVoR](https://github.com/daviddmc/NeSVoR), licensed
under MIT; its original copyright and license are included in `LICENSE`.

The bundle contains the G3R structural prior models, their architecture dependencies,
image containers and NIfTI I/O, rigid transforms, slice acquisition CUDA
kernels, and reconstruction utilities. The selected model definitions retain
their original mathematical implementations and parameter names:

- `G3RPriorV1` is the v1 residual prior, with conjugate-gradient reconstruction
  and a residual 3D refiner.
- `G3RPriorV2` is the v2 density-gated prior, whose refiner uses reconstruction
  coverage to gate its residual correction.

Other SVoRT model variants, the general NeSVoR
CLI, INR, preprocessing, datasets, and offline model training scripts are
excluded. Checkpoints are distributed separately by the G3R release.

Install a CUDA enabled PyTorch build and its matching torchvision build, plus
the packages in `requirements.txt`. Building the two extensions also requires
the CUDA toolkit (`nvcc`) and a C++ compiler supported by that toolkit.
From this directory, run:

```bash
python setup.py build_ext --inplace
```

This builds the extensions into the local `nesvor` package and does not install
or replace a global NeSVoR package. Set `PYTHONPATH` to this directory when
importing the bundle directly; the top level G3R entrypoint selects it itself.
