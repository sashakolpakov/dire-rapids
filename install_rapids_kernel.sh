#!/bin/bash
set -e

echo "=========================================="
echo "RAPIDS 25.06 Installation"
echo "=========================================="

sudo -u ec2-user -i <<'EOF'

# Install Miniforge
if [ ! -d "$HOME/miniforge3" ]; then
    echo "Installing Miniforge..."
    cd /tmp
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
    $HOME/miniforge3/bin/conda init bash
    echo "✓ Miniforge installed"
fi

source $HOME/miniforge3/etc/profile.d/conda.sh
conda config --set channel_priority flexible

# Clean
echo "Cleaning old installations..."
conda env remove -n rapids -y 2>/dev/null || true
rm -rf "$HOME/.local/share/jupyter/kernels/rapids"

# Step 1: Create base environment with Python
echo ""
echo "Step 1: Creating base environment..."
conda create -y -n rapids python=3.11 -c conda-forge

conda activate rapids

# Step 2: Install glibc 2.28+ into the environment
echo ""
echo "Step 2: Installing glibc 2.28 libraries into environment..."
conda install -y -c conda-forge sysroot_linux-64 gxx_linux-64

# Step 3: Now install RAPIDS (glibc is available in the environment)
echo ""
echo "Step 3: Installing RAPIDS 25.06..."
conda install -y -c rapidsai -c conda-forge -c nvidia \
  rapids=25.06 \
  cuda-version=12.4 \
  cupy \
  ipykernel \
  boto3 \
  sagemaker-python-sdk

# Install kernel
echo ""
echo "Step 4: Installing Jupyter kernel..."
python -m ipykernel install --user --name rapids --display-name "Python (RAPIDS 25.06)"

# Configure kernel to use conda's glibc
echo ""
echo "Step 5: Configuring kernel..."
python <<'PYEOF'
import json
import os

kernel_path = os.path.expanduser("~/.local/share/jupyter/kernels/rapids/kernel.json")
conda_prefix = os.path.expanduser("~/miniforge3/envs/rapids")

with open(kernel_path) as f:
    spec = json.load(f)

# Include sysroot lib paths where conda's glibc 2.28 lives
lib_paths = [
    f'{conda_prefix}/lib',
    f'{conda_prefix}/lib64',
    f'{conda_prefix}/x86_64-conda-linux-gnu/sysroot/lib',
    f'{conda_prefix}/x86_64-conda_cos7-linux-gnu/sysroot/lib'
]

# Filter existing paths
lib_paths = [p for p in lib_paths if os.path.isdir(p)]

spec['env'] = {
    'LD_LIBRARY_PATH': ':'.join(lib_paths),
    'CUDA_HOME': conda_prefix
}

with open(kernel_path, 'w') as f:
    json.dump(spec, f, indent=2)

print(f"✓ Kernel configured")
print(f"  LD_LIBRARY_PATH: {spec['env']['LD_LIBRARY_PATH']}")
PYEOF

# Verify with proper library path
echo ""
echo "Step 6: Verifying..."
SYSROOT=$(find $CONDA_PREFIX -type d -name sysroot 2>/dev/null | head -1)
export LD_LIBRARY_PATH="$SYSROOT/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"
python -c "import cuvs; print(f'✓ cuVS: {cuvs.__version__}')"
python -c "import cuml; print(f'✓ cuML: {cuml.__version__}')"
python -c "import cupy; print(f'✓ CuPy: {cupy.__version__}')"

echo ""
echo "✓ RAPIDS 25.06 installed successfully!"
echo "Restart Jupyter and select 'Python (RAPIDS 25.06)' kernel"

EOF
