
# coding: utf-8

# In[2]:


import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf
from utils import download, extract, get_cifar10_batches, get_cifar10_test_batch


# In[ ]:


cifar_10_download = Path("data/cifar-10-python.tar.gz")
if not cifar_10_download.exists():
    download("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", str(cifar_10_download))


# In[5]:


extract_loc = Path("data/cifar-10-batches-py")
if not extract_loc.exists():
    extract(cifar_10_download)


# In[4]:


batches = get_cifar10_batches()
test_batch = get_cifar10_test_batch()


# In[5]:


n_epochs = 10
batch_size = 64
n = 32
n_channels = 3
n_samples = batches.shape[0]


# In[6]:


inp = tf.placeholder(tf.float32, [batch_size, n, n, n_channels])
category = tf.placeholder(tf.int32, [batch_size, 1])
conv1 = tf.layers.conv2d(
    inputs=inp,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4)

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=category, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for ite in range(n_samples // batch_size):
        batch = batches.sample(n=batch_size)
        x, y = batch[1], batch[0]
        l, _ = sess.run([loss, train_op], feed_dict={inp: np.stack(list(x)).reshape([batch_size, n, n, n_channels]), category: np.stack(list(y)).reshape([batch_size, 1])})
        this_test_batch = test_batch.sample(n=batch_size)
        test_x, test_y = this_test_batch[1], this_test_batch[0]
        t_l = sess.run([loss], feed_dict={inp: np.stack(list(test_x)).reshape([batch_size, n, n, n_channels]), category: np.stack(list(test_y)).reshape([batch_size, 1])})
        print("iter: {}, train loss: {:.2f}, test loss: {}".format(ite, l, t_l))

