# GPUdecide
GPUdecide is a hardware-accelerated BDD package. There is current support for the following:

- [x] Efficiently performing quasi-reduced BDD operations on single CPU cores and on NVIDIA GPUs.
- [x] Boolean operators such as conjunction, disjunction, exclusive or, etc.
- [x] Import and export to the dddmp format.

These features are currently under development:

- [ ] Quantifiers.
- [ ] Offloading data to external memory.
- [ ] Efficient multi-core BDD operations.
- [ ] Multi-GPU BDD operations.

# Compilation instructions
The universal prerequisite for compiling the project is the availability of the CUDA development SDK. Check if it is installed using the following command:

```
nvcc --version
```

Your output should look something like this:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:57:39_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

## Python bindings
The first step in compiling the python bindings is to create a venv. 

```
python -m venv ./.venv
```

Then activate it. The following is the command for bash, for other platforms please consult the [Python website](https://docs.python.org/3/library/venv.html#how-venvs-work)

```
source ./.venv/bin/activate
```

Now install the required dependencies:

```
pip install scons pybind11
```

Finally, make sure you're in the project root and compile the project. Add ```-j[n]``` to enable multi-threading for faster compilation, by replacing ```[n]``` with your desired number of threads.

```
scons -j20
```

The python bindings are now compiled to your venv. You can access GPUdecide by writing ```import gpudecide```
