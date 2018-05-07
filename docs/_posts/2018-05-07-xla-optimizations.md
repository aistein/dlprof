---
excerpt: fusion of linear algebra operations on the GPU for massive speedup
---

## Accelerated Linea Algebra JIT Compiler Optimization
The XLA libraries available for tensorflow must be custom-compiled, and are only compatible with Nvidia compute capability 5.2 or greater (P100 or better!).  Though extremely difficult to get working, it is well worth the trouble, as these traces below demonstrate:

No XLA on the System:
![base trace]({{ "/dlprof/assets/timeline_no_system_xla.png" }})

XLA on system but diabled:
![mkl trace]({{ "/dlprof/assets/timeline_system_xla_diabled.png" }})

XLA on system enabled:
![mkl trace]({{ "/dlprof/assets/timeline_sytem_xla_enabled.png" }})
