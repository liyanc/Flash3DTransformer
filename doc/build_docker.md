# Build with Docker

## Development
One can `cd` into the repo directory and run the following script to build the Flash3D docker image for development
and deployment.

```shell
docker build -t f3d-img -f docker/Dockerfile .
```

Developers can mount the repo directory and work in interative mode through docker.

```shell
docker run --gpus all -it -v $(pwd):/workspace/Flash3D f3d-img bash
```

To build and test, one can run
```shell
# For A100 GPUs
export F3D_CUDA_ARCH=80
python3 setup.py develop
python -m unittest discover -s tests
```