# GPU programming

CERN STEAM Academy 2026 — course repository.

**Conveners / speakers:** Anton Wijs, Jan Heemstra (Eindhoven University of Technology)

## Overview

Since 2007, graphics processing units (GPUs) have been used to accelerate scientific computations other than image processing, with tremendous success. In many fields, they revolutionised the state-of-the-art. This has happened for computational biology, statistics, and physics, among others. With deep learning, GPUs have made an enormous impact in Artificial Intelligence.

GPUs offer an increasingly important alternative to the more traditional computing with Central Processing Units (CPUs). Science published an article by Leiserson et al. in 2020 [66] in which it was concluded that the computational growth of new CPUs has stagnated, and that therefore major computational advances will have to come more and more from the use of massively parallel algorithms, such as the ones running on GPUs.

GPUs have an enormous computational power, but a very restricting computational paradigm, called Single Instruction Multiple Threads (SIMT). This essentially means that computations ideally consist of instructions that can be simultaneously performed by thousands of threads, every thread applying those instructions on different data. A
good example of a computation fitting this paradigm is matrix-vector multiplication: it can be performed in parallel by assigning one thread to each matrix row. Given a row i, a thread computes the i-th entry of the resulting vector. An alternative is to assign one thread to each matrix entry; although more complicated, this is also very much in line with SIMT.

Computations for which it is less clear whether they can be accelerated effectively with GPUs are those that highly depend on the input data. Graph processing, for instance, tends to highly depend on the structure of the input graph (representing, for instance, a social network or a road map). Still, impressive acceleration has been achieved
by researchers over the years.

In this course, we will take a close look at GPU programming, from the basic principles to very involved low-level optimisations. Furthermore, we will take an in-depth look at various real-world applications in which GPU-acceleration has been effectively applied, and discuss the algorithms that have enabled these successes, which are in some cases vastly different from the state-of-the-art algorithms for CPU computing. Graph search, symbolic reasoning with Binary Decision Diagrams, and formal verification are some of the applications that will be presented. In a number of lab sessions, the course participants will get hands-on experience accelerating some of these algorithms, such as parallel reduction, matrix operations, and graph search.

## Environment setup

This course runs on the academy laptops (AlmaLinux 9). For the shared environment —
accounts, WiFi, editors, lxplus, AFS, CERNBox — see the
[STEAM Academy documentation](https://stac.docs.cern.ch/). In particular, see (https://stac.docs.cern.ch/ngt/gpu-resources/) for setting up a JupyterLab notebook with GPU access.

_TODO: This course requires CUDA C++. See below how to obtain the code for all lab sessions, and how to work with the files of a lab session:_

```sh
# get the code
git clone https://github.com/CERN-STEAM-Academy/26-GPU-PROGRAMMING.git
cd 26-GPU-PROGRAMMING

# go to the files of a particular lab session
cd labnumber-name

# compile the source code of the exercise
make

# run the resulting executable
./exercise-name
```

## Materials

The slides will be made available shortly in the [a relative link](slides) folder.

## Schedule

See the [Indico timetable](https://indico.cern.ch/event/1697464/timetable).
