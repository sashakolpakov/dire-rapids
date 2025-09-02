from setuptools import setup, find_packages
import os

setup(
    name="dire-rapids",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pykeops",
        "loguru",
        "tqdm",
    ],
    extras_require={
        "cuvs": [
            "rapids-25.08",
            "cuml",
            "cuvs",
            "cupy",
        ],
        "dev": [
            "pytest",
            "matplotlib",
            "scikit-learn",
        ],
    },
    python_requires=">=3.8",
    author="Sasha Kolpakov",
    description="PyTorch and RAPIDS accelerated dimensionality reduction",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/sashakolpakov/dire-rapids",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)