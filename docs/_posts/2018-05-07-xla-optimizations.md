---
excerpt: fusion of linear algebra operations on the GPU for massive speedup
---

## Accelerated Linea Algebra JIT Compiler Optimization
The XLA compiler available for tensorflow must be custom-built from source, and is only compatible with Nvidia devices with compute capability 5.2 or greater (P100 or better!).  Though extremely difficult to get working, it is well worth the trouble, as these traces below demonstrate:

**No XLA on the System:**
![base trace]({{ "/dlprof/assets/timeline_no_system_xla.png" }})

**XLA on system but diabled:**
![mkl trace]({{ "/dlprof/assets/timeline_system_xla_diabled.png" }})

**XLA on system enabled:**
![mkl trace]({{ "/dlprof/assets/timeline_sytem_xla_enabled.png" }})

### Background
Like many compilers, XLA produces an intermediate representation (IR) from the original tensorflow code, with which it analyzes whether or not certain linear algebra calculations can be fused together to execute more efficiently.  When executed well, this methodology can significantly reduce running time on CPU and GPU, reduce memory usage, and enable cross-platform compatibility. Tensorflow has published a series of posts detailing the techincal details of XLA [here](https://www.tensorflow.org/performance/xla/).

### Setup Guide
What follows is a step-by-step guide to build, install, and test the Tensorflow XLA JIT compiler. Unfortunately, since the software is still in experimental development phase, we were only able to demonstrate its use with some very specific system settings. Briefly, this process involves taking an existing [docker container](https://ngc.nvidia.com/registry/nvidia-tensorflow) from an Nvidia managed image repository, updating its configurations so that we may rebuild tensorflow from scratch on it, and pip-installing the generated .whl.  Testing, as in previous posts, all takes place within the safety of the docker container so as not to contaminate the host system's settings. 

  - Hardware Configuration
    - Gcloud Copmute instance
    - 8x vCPUs, 30G RAM
    - 1x Nvidia Tesla P100 GPU
    - 256G SSD Drive
  - System Requirements
    - Ubuntu 16.04 LTS
    - Python3.6, with Miniconda package manager
    - Nvidia-docker2
