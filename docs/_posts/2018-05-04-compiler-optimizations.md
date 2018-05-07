---
excerpt: the uncertain and treacherous path of building tensorflow from scratch
---

## Tensorflow Compiler Optimizations

---
*Difficulty: Intermediate*

---
The Tensorflow team's prime objective is to spread the use of its framework over as broad an audience as possible.  There are many classes of users who wish to take advantage of deep learning, from students to researchers to professional engineering teams.  It makes sense that by default Tensorflow has a standard set of configurations because not every use-case requires optimization.  However, the advanced users will be interested in fine-tuning the software to meet their needs.  Unfortunately, the instructions provided by Tensorflow towards this end are rather sparse.

## Overview
In this tutorial - which we have adapted from a fabulous series of [posts](https://www.pugetsystems.com/labs/hpc/Build-TensorFlow-CPU-with-MKL-and-Anaconda-Python-3-6-using-a-Docker-Container-1133/) by Dr. Donald Kinghorn at Puget Systems - we will walk through compiling Tensorflow from scratch three times: (1) Using the Intel MKL CPU optimizations; (2) Enabling the accelerated linear algebra (XLA) framework; (3) Combining these optimizations with GPU capability.  We will be using Docker containers to insulate our systems from the mess of compilation.  The result in each case will be a .whl file which can be pip-installed and tested inside of that Docker container.  Should you be happy with the results, you can then take the .whl file and install it the same way on your host system!

### Baseline: Compiling TF for CPU with No Optimizations
In order to extract meaningful results from this exercise, we must be able to show the difference in performance between builds.  Despite the fact we aren't adding any special compile switches here, it is still important for us to do this build with "docker isolation" because we must be careful not to cross-contaminate libraries between tensorflow installations.  We tried to test outside of docker initially and found that all tests yeilded the same results because anaconda was cacheing tensorflow libraries for reuse between different conda environments!  Here instead we do the build and test in the safe confines of a Docker container.

Before you begin, please [install and configure Docker](https://www.pugetsystems.com/labs/hpc/How-To-Setup-NVIDIA-Docker-and-NGC-Registry-on-your-Workstation---Part-1-Introduction-and-Base-System-Setup-1095/). These instructions are copied from Dr. Kinghorn's post mentioned above, reformatted here for convenience.

1. Make a directory to do your build
```bash
$ mkdir TF-build
$ cd TF-build
```
2. Download tensorflow source code and checkout version 1.7
```bash
$ git clone https://github.com/tensorflow/tensorflow
$ cd tensorflow/
$ git checkout r1.7
```
3. Setup docker container build directory
```bash
$ mkdir dockerfile
$ cd dockerfile
```
4. Supply the necessary dependency files/hosts for Anaconda and Bazel. Note that if you are using another system aside from x86-linux you will need to acquire the appropriate anaconda file.
```bash
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" > bazel.list
```
5. Create the Dockerfile. Save the following as "Dockerfile", with a capital D!

    ```bash
    # Dockerfile to setup a build environment for TensorFlow
    # using Intel MKL and Anaconda3 Python
    
    FROM ubuntu:16.04
    
    # Add a few needed packages to the base Ubuntu 16.04
    # Dr. Kinghorn prefers emacs-nox, We prefer vim-nox
    RUN \
        apt-get update && apt-get install -y \
        build-essential \
        curl \
        vim-nox \
        git \
        openjdk-8-jdk \
        && rm -rf /var/lib/lists/*
    
    # Add the repo for bazel and install it.
    # I just put it in a file bazel.list and coped in the file
    # containing the following line
    # deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8
    COPY bazel.list /etc/apt/sources.list.d/
    RUN \
      curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
      apt-get update && apt-get install -y bazel
    
    # Copy in and install Anaconda3 from the shell archive
    # Anaconda3-5.1.0-Linux-x86_64.sh
    COPY Anaconda3* /root/
    RUN \
      cd /root; chmod 755 Anaconda3*.sh && \
      ./Anaconda3*.sh -b && \
      echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> .bashrc && \
      rm -f Anaconda3*.sh
    
    # That's it! That should be enough to do a TensorFlow 1.7 CPU build
    # using Anaconda Python 3.6 Intel MKL with gcc 5.4
    ```
    
6. Create the Docker container and run it. Note you will have to set the environmental variable PROJECT yourself to the proper path to your working directory. **Note:** you must always be in the *dockerfile* directory to use the local configurations file.
```bash
$ docker build -t tf-build-1.7-cpu-mkl-only .
$ docker run --rm -it -v $PROJECT/TF-build:/root/TF-build tf-build-1.7-cpu-mkl-only
```
  - A quick aside about Docker and the command line arguments here, because they are interesting.  If we consider a "virtual machine" as abstracting the *hardware* so that any *operating system* may run upon it, we can similarly consider a "container" as abstracting the *operating system* so that any *application* may run upon it. As such, the most common use-cases for containers do not require much if any interaction with the container via command line, usually only through whatever interface is provided by the application.  In the case of this tutorial, however, we are using Docker more like a virtual machine than a a container.  We'll be using the command line to interact with it, just like we would in a VM. The difference here is that a container is much lighter-weight (and much less capable).  It will build and be ready for use in minutes, and there's no downloading multi-gigabyte .iso files necessary.  Once we are done doing the build, we'll blow it away.  Note the significance of the command line options for ```docker run``` used ([documentation here](https://docs.docker.com/engine/reference/commandline/run/#options)):
    - ```--rm``` will delete the container from our system once we exit the instance
    - ```-i``` keep stdin open so that the container may receive our input
    - ```-t``` allocates a pseudo-TTY (text-only console)
    - ```-v``` mount the *volume* (folder containing tensorflow source-code) at a specific point inside the container
7. Configure Tensorflow. You should now be greeted with a custom CLI prompt, which indicates that we are running inside the container.
```bash
> cd root/TF-build/tensorflow
> ./configure
```
  - Say yes to "jemalloc support", and no to every other prompt (including CUDA support, as we are not yet demonstrating GPU).
8. Build Tensorflow. **Warning:** This can take quite some time, on the order of 30 minutes in the case of our GCP instance.
```bash
> bazel build //tensorflow/tools/pip_package:build_pip_package
```
9. Create the pip package
```bash
> bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tensorflow_pkg
```

We test this base installation in a new conda environment within the docker container:
```bash
> conda create --name tf-cpu-base
> source activate tf-cpu-base
> pip install tensorflow_pkg/tensorflow-1.7.1-cp36-cp36m-linux_x86_64.whl
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
2018-05-04 16:51:19.722377: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
-873849.9
 took 22.90920376777649 seconds
 ```

A more in depth performance comparison is the subject of a later section in this post, but let us briefly inspect the output.  The given information states that we have certain capabilities on our CPU which are not being utilized by our Tensorflow build: SSE4.1 SSE4.2 AVX AVX2 FMA.  What are they?
- **SSE4 Instructions:** "Streaming SIMD Extensions 4". These are assembly instructions for Intel and AMD processors which allow for "packed" read/writes, string comparisons, and integer operations.
- **AVX Instructions:** "Adanced Vector Extensions" allow Intel and AMD processors to do mathematical operations and memory manipulations on up to 256 bits of input data at a time.
- **FMA Instrucions:** "Fused-Multiply Accumulate" instructuions are exactly as the name implies: In a single basic computational step, Intel and AMD processors with these extensions can - for example - take 3 inputs a,b,c and produce a = a*c + b.
It is clear that having these instruction sets enabled in our Tensorflow build would improve the performance of any program that could benefit from SIMD (single-instruction multiple-data), and matrix multiplication is exactly one such application!

Now you may be thinking "though these instructions can help with SIMD on the CPU, why should we even bother? Afterall, isn't SIMD exactly what GPGPU is for?!"  That is a very good question without a straightforward answer, and its discussion is certainly beyond the scope of this post.  For a thorough understanding of the complexities of this question, check out this (somewhat outdated) white-paper by Intel, ["Debunking the 100x GPU vs. CPU Myth"](http://sbel.wisc.edu/Courses/ME964/Literature/LeeDebunkGPU2010.pdf).

### Optimization 1: Compiling with Intel MKL Libraries
In order to enable Tensorflow to use SSE4, AVX, and FMA instructions, we must compile it from the source code with the special siwtch ```--config=mkl```. The steps to do this are exactly the same above, but replacing step 8 with the following:

8. Build Tensorflow. **Warning:** This can take quite some time, on the order of 30 minutes in the case of our GCP instance.
```bash
> bazel build --config=opt --config=mkl //tensorflow/tools/pip_package:build_pip_package
```

That's it! Now it is time to see what kind of performance we gained from the scratch compilation.

```bash
> python testMM.py
```
Here is the output we got:
```bash
-873847.3
 took 9.988160133361816 seconds
```

Fantastic! The warning about SSE4, AVX, and FMA capabilities has disappeared, and our matrix multiplication took less than half the original time!  For a better understanding of what changed, we built some profiles. Upon deploying the native python profiler, via ```python -m cProfile -s cumtime mm_test.py &> profile.txt``` we found the profiles very hard to interpret.  As such, we decided to use the native tensorflow chrome-trace to get more insight:
```python
import tensorflow as tf
import time
from tensorflow.python.client import timeline

tf.set_random_seed(42)
A = tf.random_normal([10000,10000])
B = tf.random_normal([10000,10000])
def checkMM():
        start_time = time.time()
        with tf.Session() as sess:
                # options to trace execution
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                print( sess.run( tf.reduce_sum( tf.matmul(A,B) ),\
                                 options=options,\
                                 run_metadata=run_metadata ) )

                # create timeline object and write to json
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                        f.write(chrome_trace)

        print(" took {} seconds".format(time.time() - start_time))
checkMM()
```
In your Google Chrome browser, you can view the output file ```timeline.json``` by navigating to ```chrome://tracing``` and then loading the json file.

Base-Trace:
![base trace]({{ "/dlprof/assets/base-trace.png" }})

MKL-Trace:
![mkl trace]({{ "/dlprof/assets/mkl-trace.png" }})

#### Installation on the Host System
If you're happy with the results, then install this tensorflow build on your local system! Since we mounted the volume(s) ```TF-Build*``` into the container during each of the above tests, the .whl files are saved on the host system under ```$TF-Build*/tensorflow_pkg/```.

```bash
$ conda create tf-cpu-mkl-only
$ source activate tf-cpu-mkl-only
$ cd ../../
(tf-cpu-mkl-only) $ pip install tensorflow_pkg/tensorflow-1.7.1-cp36-cp36m-linux_x86_64.whl
```
