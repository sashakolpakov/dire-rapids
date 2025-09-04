from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="dire-rapids",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "benchmarking", "benchmarking.*"]),
    
    # Core dependencies
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "pykeops>=2.1.0",
        "loguru>=0.6.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "plotly>=5.0.0",
    ],

    # Python versioning
    python_requires=">=3.8",
    
    # Optional dependencies
    extras_require={
        "cuda": [
            "cupy-cuda12x>=10.0.0",  # Generic CUDA 12.x support
        ],
        "rapids": [
            "cuml-cu12>=23.0.0",
            "cuvs-cu12>=23.0.0",
            "cudf-cu12>=23.0.0",
            "cupy-cuda12x>=10.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-timeout>=2.1.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pylint>=2.15.0",
            "mypy>=0.950",
            "memory_profiler>=0.60.0",
            "line_profiler>=3.5.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-timeout>=2.1.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "myst-parser>=0.17.0",
        ],
    },
    
    # Metadata
    author="Alexander Kolpakov (UATX), Igor Rivin (Temple University)",
    author_email="akolpakov@uaustin.org, rivin@temple.edu",
    maintainer="Alexander Kolpakov",
    maintainer_email="akolpakov@uaustin.org",
    description="PyTorch and RAPIDS (cuVS/cuML) accelerated dimensionality reduction",
    long_description=read_readme() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/sashakolpakov/dire-rapids",
    project_urls={
        "Bug Reports": "https://github.com/sashakolpakov/dire-rapids/issues",
        "Source": "https://github.com/sashakolpakov/dire-rapids",
        "Documentation": "https://github.com/sashakolpakov/dire-rapids#readme",
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    
    # Keywords for PyPI
    keywords="dimensionality-reduction machine-learning gpu cuda rapids pytorch visualization embedding umap tsne",
    
    # License
    license="MIT",
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
    
    # Entry points (if you have CLI tools)
    entry_points={
        "console_scripts": [
            # Example: "dire-rapids=dire_rapids.cli:main",
        ],
    },
)
