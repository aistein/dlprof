---
excerpt: the uncertain and treacherous path of building tensorflow from scratch
---

## Why Build from Scratch?
The Tensorflow team's prime objective is to spread the use of its framework over as broad an audience as possible.  There are many classes of users who wish to take advantage of deep learning, from students to researchers to professional engineering teams.  It makes sense that by default Tensorflow has a standard set of configurations because not every use-case requires optimization.  However, the advanced users will be interested in fine-tuning the software to meet their needs.  Unfortunately, the instructions provided by Tensorflow towards this end are rather sparse.

### Overview
In this tutorial - which we have adapted from a fabulous series of [posts](https://www.pugetsystems.com/labs/hpc/Build-TensorFlow-CPU-with-MKL-and-Anaconda-Python-3-6-using-a-Docker-Container-1133/) by Dr. Donald Kinghorn at Puget Systems - we will walk through compiling Tensorflow from scratch three times: (1) Using the Intel MKL CPU optimizations; (2) Enabling the accelerated linear algebra (XLA) framework; (3) Combining these optimizations with GPU capability.  We will be using Docker containers to insulate our systems from the mess of compilation.  The result in each case will be a .whl file which can be conda-installed in a new environment.  **Important:** It is expected that you have Anaconda3 with Python3.6 on your system.

### Baseline: Compiling TF for CPU with No Optimizations
In order to extract meaningful results from this exercise, we must be able to show the difference in performance between builds.  Having a baseline in which no optimizations were applied will enable us to do this. Since we are happy with default conigurations for the baseline, it is not necessary to take the extra step of "docker isolation" -- a direct conda install is sufficient.

The base installation in a new conda environment is simple:
```bash
$ conda create --name tf-cpu-base tensorflow
$ source activate tf-cpu-base
```
A quick test of matrix multiplication will elucidate our motivation for the undertaking ahead.  Running the following python script
```python
import tensorflow as tf
import time
tf.set_random_seed(42)
A = tf.random_normal([10000,10000])
B = tf.random_normal([10000,10000])
def checkMM():
     start_time = time.time()
     with tf.Session() as sess:
             print( sess.run( tf.reduce_sum( tf.matmul(A,B) ) ) )
     print(" took {} seconds".format(time.time() - start_time))
checkMM()
```
produces the following results:
```bash
2018-05-04 16:51:19.722377: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
-873844.7
 took 7.3969504833221436 seconds
 ```

A more in depth performance comparison is the subject of a later section in this post, but let us briefly inspect the output.  The given information states that we have certain capabilities on our CPU which are not being utilized by our Tensorflow build: SSE4.1 SSE4.2 AVX AVX2 FMA.  What are they?
**SSE4 Instructions:** "Streaming SIMD Extensions 4". These are assembly instructions for Intel and AMD processors which allow for "packed" read/writes, string comparisons, and integer operations.
**AVX Instructions:** "Adanced Vector Extensions" allow Intel and AMD processors to do mathematical operations and memory manipulations on up to 256 bits of input data at a time.
**FMA Instrucions:** "Fused-Multiply Accumulate" instructuions are exactly as the name implies: In a single basic computational step, Intel and AMD processors with these extensions can - for example - take 3 inputs a,b,c and produce a = a*c + b.
It is clear that having these instruction sets enabled in our Tensorflow build would improve the performance of any program that could benefit from SIMD (single-instruction multiple-data), and matrix multiplication is exactly one such application!
