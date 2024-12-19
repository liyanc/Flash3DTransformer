# Profile with [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

We provide easy interfaces to instrument and profile Flash3D through NVIDIA Nsight Systems.

## Tracing a program
One can easily trace a python program or even a training session in two lines of commands with our helpers.

```shell
# Import our helpers
source profiling/instrumentation.sh
# Assume the working directory is `/workspace/traces`
export WORKDIR=/workspace/traces

# Trace a program, and convert to sqlite
profile_nsys "$WORKDIR"/buckswin_attn.nsys-rep python3 -m unittest tests/test_attn_fwd.py
nsys_to_sqlite "$WORKDIR"/buckswin_attn.nsys-rep "$WORKDIR"/buckswin_attn.sqlite
```

Now you have a complete database `buckswin_attn.sqlite` for performance events throughout the program.
One can perform standard SQL queries on the database to gather metrics of interest.

We also include SQL scripts to reproduce the plots in our paper.