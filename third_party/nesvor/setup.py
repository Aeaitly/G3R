"""Build G3R's vendored NeSVoR CUDA extensions in this source tree."""

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
version = {}
exec((ROOT / "nesvor" / "version.py").read_text(), version)


def extension(name, directory, stem):
    return CUDAExtension(
        name=f"nesvor.{name}",
        sources=[
            str(ROOT / "nesvor" / directory / f"{stem}.cpp"),
            str(ROOT / "nesvor" / directory / f"{stem}_kernel.cu"),
        ],
    )


setup(
    name="nesvor-g3r-inference",
    version=version["__version__"],
    description="NeSVoR inference utilities for G3R",
    url=version["__url__"],
    author=version["__author__"],
    license="MIT",
    packages=find_packages(),
    package_data={
        "nesvor": [
            "py.typed",
            "transform/*.cpp",
            "transform/*.cu",
            "slice_acquisition/*.cpp",
            "slice_acquisition/*.cu",
        ]
    },
    ext_modules=[
        extension("slice_acq_cuda", "slice_acquisition", "slice_acq_cuda"),
        extension("transform_convert_cuda", "transform", "transform_convert_cuda"),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
