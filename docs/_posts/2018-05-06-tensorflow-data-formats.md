---
excerpt: Tensorflow claims that when using nvidia gpus, the data format NCHW is more performant than NHWC for convolutional networks. We use the Tensorflow profiler to show this is the case.
---

## Data Formats

One of the proposed optimizations according to Tensorflow is the choice of data format NCHW when using Nvidia GPUs. Here we will demonstrate and explain why.

To begin, we need to understand just what the acronym NCHW is. Specifically, in the context of convolutional networks and max pooling layers, NCHW is an acronym for the dimensions of the data being represented.

1. N - number of items in the batch
1. C - number of channels
1. H - height of the matrix in each channel
1. W - width of the matrix in each channel

So, clearly, if we use NCHW format and have input dimensions `(32, 3, 28, 28)`, our data has a batch size of 32 and 3 input channels each of which is a 28 x 28 dimension matrix.

The alternative is NHWC, also known as `channels_last` in some documentation, whihc simply places the channel dimension at the last value.

## Tensorflow Data Formats

Tensorflow has an unfortunate history with data formats. Originally, Tensorflow was only compatible with NHWC because it was slightly faster on CPUs. As we will explain, Tensorflow later introduced support for NCHW as Nvidia was able to optimize computations for this format.

## Experiments

In order to conduct experiments we needed two things

1. A model that uses 2D convolutions
1. A way to measure the running time of convolutions independent of other operations

Satisfying (1) was much simpler than satisfying (2), but both required that we clone the tensorflow repository. 

### A model

Within the Tensorflow repository, under the `tensorflow/tensorflow/examples/tutorials/mnist/` directory is a project `mnist_deep.py`. We used this as the base to our model because it involves 2 layers which both use 2D convolutions along with max pooling. In order to easily run our model with either data format, we simply added an argument `data_format` to specify, then built our computational graph accordingly.

```bash
python mnist_deep.py --data_format NCHW
```

### Measuring Performance

In order to confirm that any improvement in performance is actually due to an improvement in convolutional computation speed, we needed to set up Tensorflow's profiling tool `tfprof`.

For this particular case, it was a fairly painless experience, but I expect that to be the exception and not the rule. We simply had to cd into the directory in the Tensorflow repo, `tensorflow/tensorflow/core/profiler/` and run one command.

```bash
bazel build .
```

The entire build process took approximately 40 minutes, but also included several other Tensorflow tools (which makes me think cd\-ing into the directory is not necessary).

Once the build was complete we had a file linked to an executable in the root directory of our project under `bazel-bin/tensorflow/core/profiler/profiler`.

Now that the profiler command line tool was installed we needed the model to generate profiles that could be parsed. This was another very easy addition into the original model.

```python
  with tf.contrib.tfprof.ProfileContext(profile_dir) as pctx:
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # profiler = tf.profiler.Profiler(sess.graph)
      # time_per_epoch = []
      for i in range(2000):
        batch = mnist.train.next_batch(50)
        ...
```

The only line we needed to add was `with tf.contrib.tfprof.ProfileContext(...) as pctx`. After adding this line, Tensorflow's profiler will randomly sample steps of your training loop and dump the resulting profile to the specified directory.

## Results

As stated in the Tensorflow performance guide, our convolutional operations performed better using the NCHW format than with the NHWC format. The output from the profiler after 2000 iterations was as follows

### NCHW

```bash
Profile:
node name                      | requested bytes               | total execution time          | accelerator execution time      | cpu execution time        | op occurrence (run|defined)
...
Conv2DBackpropFilter           89.20MB (95.93%, 26.08%),        1.64ms (67.76%, 14.83%),        1.42ms (63.40%, 17.18%),          219us (80.78%, 7.91%),        2|2
Conv2D                         93.78MB (69.84%, 27.42%),        1.50ms (52.94%, 13.56%),        1.22ms (46.22%, 14.79%),          275us (72.87%, 9.93%),        2|2
Conv2DBackpropInput            95.94MB (42.42%, 28.05%),        1.16ms (39.37%, 10.44%),        1.03ms (31.43%, 12.39%),          129us (62.93%, 4.66%),        1|2
...
MaxPoolGrad                      7.53MB (14.36%, 2.20%),          383us (20.57%, 3.46%),          303us (11.71%, 3.66%),           79us (47.00%, 2.85%),        2|2
MaxPool                          1.88MB (12.16%, 0.55%),          199us (12.56%, 1.80%),           123us (4.83%, 1.49%),           75us (35.59%, 2.71%),        2|2
...
```

### NHWC

```bash
Profile:
node name                     | requested bytes                | total execution time          | accelerator execution time      | cpu execution time        | op occurrence (run|defined)
...
Conv2DBackpropFilter          116.99MB (96.64%, 28.23%),        1.89ms (84.00%, 15.71%),        1.61ms (80.68%, 18.02%),          278us (93.60%, 9.08%),        2|2
Conv2D                         95.23MB (68.41%, 22.98%),        1.75ms (68.29%, 14.55%),        1.42ms (62.67%, 15.85%),         332us (84.52%, 10.84%),        2|2
Conv2DBackpropInput           100.96MB (45.42%, 24.36%),        1.27ms (39.94%, 10.59%),        1.11ms (32.22%, 12.49%),          156us (62.19%, 5.09%),        1|2
MaxPoolGrad                    45.64MB (21.06%, 11.01%),          906us (29.35%, 7.54%),          719us (19.73%, 8.05%),          186us (57.10%, 6.07%),        2|2
...
MaxPool                          1.88MB (10.04%, 0.45%),          237us (12.17%, 1.97%),           180us (5.12%, 2.02%),           56us (32.68%, 1.83%),        2|2
...
```

As we can see from the `total execution time` column every operation related to convolutions, includeing max pooling, performed better. It is important to note that these results were consistent, but we are only displaying the result of one profile over relatively few iterations. Any larger model like those used in modern object recognition or segmentation problems would have much more tangible improvements.

## Explanation

So the question remains, what is happening under the hood that makes Nvidia GPUs run convolutions faster in one format than in another?

As we know, Nvidia's cuDNN library is written using the CUDA platform, allowing users to write in languages like c/c++. Because these languages access data in a row\-major fashion, they can simply slice the NCHW tensor on `(1, 2)` to get the first image's second channel. If the data were instead formatted as NHWC they would have to slice along some axis like `(1, :, :, 2)` to get the same data. This leads to a slight increase in complexity when operating on NHWC data.

According to the Tensorflow team running convolutions on CPUs is slightly more performant when the data is NHWC format. Additionally, this problem could be a hold\-over from other languages like Matlab, which access matrices in a column\-major fashion. In those languages, the performance in accessing data is the opposite.

Unfortunately NHWC is still the default data format for Tensorflow, and we found that the NCHW format threw exceptions for some operations when run on CPU (`max_pooling`)
