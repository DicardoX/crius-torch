# Crius Infrastructure with Torch-based Implementation

## Installation

To build Docker images and run container:

```bash
# Build image (add `--network=host` if `Temporary failure resolving XXX`)
docker build -t dicardo/crius-torch:0.0-dev .
# Run container
docker run --runtime=nvidia -d -it --rm --gpus all --shm-size 64g --network=host --privileged --volume /home/cyxue/Projects/crius-torch/.cache:/root/.cache --env NVIDIA_DISABLE_REQUIRE=1 --name crius-torch-dev dicardo/crius-torch:0.0-dev
```

To install the editable alpha version of crius from source:

```bash
git clone --recursive https://github.com/DicardoX/crius-torch.git
conda activate crius
pip install -e .
# Check installation
python -c "import crius;crius.check_installation()"
```
