# Unit Tests

Many of our unit tests depend on real data from the KITTI dataset.
Before running unit tests, please setup the KITTI dataset directory and declare the **environment variable** pointing
to the directory.

```shell
export KITTI_RT="path/to/kitti"
```

In addition, our unit tests would run on the `cuda:0` device.
If you prefer alternative devices, please set the environment variable
`CUDA_VISIBLE_DEVICES={device_of_your_choice}` before running unit tests.

## Run All Tests
One can run all unit tests in the repo root.
```shell
cd {flash3d root}
python -m unittest discover -s tests
```