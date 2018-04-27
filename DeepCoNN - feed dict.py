
# coding: utf-8

# In[1]:


import datetime
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import time

from tensorflow.python import debug as tf_debug


# In[2]:


truncate_len = 800
batch_size = 32


# In[3]:


action_movie_lover = """i really love action movies they are my favorite kind of movie because i love to watch the good guys
win. some of my favorite actors are jet li and jackie chan because they were the last great actors who
actually knew how to fight. modern action movie actors are just pretty faces and the editors swap camera
angles when supposed hits make contact"""
action_movie_hater = """I really hate action movies. jackie chan and jet li are the worst. their old fashioned
special effects, if you can even call them that, are out dated and boring. why are people even still paying
for them to make movies"""
item = """this movie is great because of the way the strong, clever, hero saves the day at the end. as always
jackie chans action directing and stunts are amazing, i'm so glad he doesn't use a stunt double liek
some other actors that don't need to be mentioned"""


# In[4]:


class Batch_Dataset(object):
    def __init__(self, user_review_location, item_review_location, rating_location, pad_length, pad_value, batch_size):
        user_review_list = [line.split()[:truncate_len] for line in self._get_lines(user_review_location)]
        self.user_review_list = np.array(self._pad_if_necessary(user_review_list, pad_value, pad_length))
        
        item_review_list = [line.split()[:truncate_len] for line in self._get_lines(item_review_location)]
        self.item_review_list = np.array(self._pad_if_necessary(item_review_list, pad_value, pad_length))
        
        self.ratings = np.array([float(rating) for rating in self._get_lines(rating_location)])
        
        self.batch_size = batch_size
        self.iter = 0
        self.stop_iter = len(self.ratings) / batch_size
    
    def _get_lines(self, fname):
        with open(fname, "rt") as data:
            return data.read().splitlines()
        
    def _pad_if_necessary(self, list_of_lists, pad_value, pad_length):
        return [np.array(line + [pad_value] * (pad_length - len(line))) for line in list_of_lists]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.iter += 1
        if self.iter > self.stop_iter:
            self.iter = 0
            raise StopIteration
        else:
            rand_indices = np.random.choice(range(len(self.ratings)), size=self.batch_size, replace=False)
            return (np.array(self.user_review_list[rand_indices]),
                    np.array(self.item_review_list[rand_indices]),
                    np.array(self.ratings[rand_indices]).reshape(self.batch_size, 1))


# In[5]:


emb_size = 50
filters = 10
kernel_size = 3
n_epochs = 824

s = time.time()

tf.reset_default_graph()

with open("data/dictionary.pkl", "rb") as f:
    dictionary = pkl.load(f)

values = list(range(len(dictionary)))
keys = list(dictionary)

table = tf.contrib.lookup.HashTable(
  tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
)

word_embeddings = tf.get_variable(
    "word_embeddings",
    shape=[len(dictionary), emb_size]
)

u_inputs = tf.placeholder(tf.string, (batch_size, truncate_len), name="user_inputs")
i_inputs = tf.placeholder(tf.string, (batch_size, truncate_len), name="item_inputs")
ratings_input = tf.placeholder(tf.float64, (batch_size, 1), name="ratings")

u_inputs_indices = table.lookup(u_inputs)
i_inputs_indices = table.lookup(i_inputs)

u_inputs_embedded = tf.nn.embedding_lookup(word_embeddings, u_inputs_indices)
i_inputs_embedded = tf.nn.embedding_lookup(word_embeddings, i_inputs_indices)

user_conv1 = tf.layers.conv1d(
    u_inputs_embedded,
    filters,
    kernel_size,
    use_bias=True,
    activation=tf.nn.tanh,
    name="user_conv")

item_conv1 = tf.layers.conv1d(
    i_inputs_embedded,
    filters,
    kernel_size,
    use_bias=True,
    activation=tf.nn.tanh,
    name="item_conv")

user_max_pool1 = tf.layers.max_pooling1d(user_conv1, 2, 1)
item_max_pool1 = tf.layers.max_pooling1d(item_conv1, 2, 1)

user_flat = tf.layers.flatten(user_max_pool1)
item_flat = tf.layers.flatten(item_max_pool1)

user_dense = tf.layers.dense(user_flat, 64, activation=tf.nn.relu)
item_dense = tf.layers.dense(item_flat, 64, activation=tf.nn.relu)

predictions = tf.reduce_sum( tf.multiply( user_dense, item_dense ), 1, keep_dims=True )

loss = tf.losses.mean_squared_error(ratings_input, predictions)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

train_op = optimizer.minimize(
    loss=loss)

vars_init = tf.global_variables_initializer()
tables_init = tf.tables_initializer()


with tf.Session() as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(vars_init)
    sess.run(tables_init)
    
    dataset = Batch_Dataset("data/train_u_reviews.txt",
                            "data/train_i_reviews.txt",
                            "data/train_ratings.txt",
                            truncate_len,
                            "unk",
                            batch_size)
    
#     i = 0
    for user_batch, item_batch, rating_batch in dataset:
        
#         i += 1
        _, l = sess.run([train_op, loss], feed_dict={
            u_inputs: user_batch,
            i_inputs: item_batch,
            ratings_input: rating_batch
        })
#         if i % 101 == 0:
#             print("{}: epoch {}, loss {:.2f}".format(str(datetime.datetime.now()), i, l))
            
e = time.time()

print("ran {} seconds".format(e - s))

