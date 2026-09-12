# G3R: Gaussian-based Geometry-Guided Reconstruction for Fetal Brain MRI

## Overview

G3R is a geometry-guided Gaussian reconstruction framework for recovering high-resolution 3D fetal brain MRI volumes from motion-corrupted 2D slices. It is designed to retain the fast reconstruction speed of Gaussian-based slice-to-volume reconstruction while improving structural stability in under-determined settings, including sparse-orientation and single-stack acquisitions.

## Motivation

Optimizing independent Gaussian primitives directly from thick, motion-corrupted slices can lead to flattened or stretched primitives, structural distortion, and projection artifacts. G3R introduces an efficient geometric prior to stabilize this optimization without relying on computationally expensive generative models.

## Method

G3R contains two main components:

- **Density-Gated Prior Network (DGPN):** estimates a structurally continuous prior volume from a physics-derived density map that describes spatial observation confidence.
- **Two-stage Gaussian reconstruction:** initializes Gaussian primitives from the prior volume, then refines them against the acquired slices using an annealed distillation constraint.

## Results

In the experiments reported in the paper, G3R retains the reconstruction speed of existing Gaussian frameworks while reaching 24.56 dB PSNR on the simulated dataset and improving PSNR by up to 1.87 dB in a single-stack stress test.

## Authors

Zhibao Cai, Yao Lv, Xin Zhang, and Chaoxiang Yang

## Code

Code is coming soon.

## Citation

Citation information will be provided with the public paper release.
