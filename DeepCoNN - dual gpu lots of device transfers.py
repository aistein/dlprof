
# coding: utf-8

# In[1]:


import pandas as pd
import pickle as pkl
import tensorflow as tf
import time


# In[2]:


def model_fn(features, labels, mode):
    emb_size = 50
    filters=10
    kernel_size=3
    
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
    with tf.device("/gpu:0"):
        u_inputs = features[0]
    with tf.device("/gpu:1"):
        u_inputs = table.lookup(u_inputs)
    with tf.device("/gpu:0"):
        u_inputs = tf.nn.embedding_lookup(word_embeddings, u_inputs)
    with tf.device("/gpu:1"):
        user_conv1 = tf.layers.conv1d(
            u_inputs,
            filters,
            kernel_size,
            use_bias=True,
            activation=tf.nn.tanh,
            name="user_conv")
    with tf.device("/gpu:0"):
        user_max_pool1 = tf.layers.max_pooling1d(user_conv1, 2, 1)
    with tf.device("/gpu:1"):
        user_flat = tf.layers.flatten(user_max_pool1)
    with tf.device("/gpu:0"):
        user_dense = tf.layers.dense(user_flat, 64, activation=tf.nn.relu)
    
    with tf.device("/gpu:1"):
        i_inputs = features[1]
    with tf.device("/gpu:0"):
        i_inputs = table.lookup(i_inputs)
    with tf.device("/gpu:1"):
        i_inputs = tf.nn.embedding_lookup(word_embeddings, i_inputs)
    with tf.device("/gpu:0"):
        item_conv1 = tf.layers.conv1d(
            i_inputs,
            filters,
            kernel_size,
            use_bias=True,
            activation=tf.nn.tanh,
            name="item_conv")
    with tf.device("/gpu:1"):
        item_max_pool1 = tf.layers.max_pooling1d(item_conv1, 2, 1)
    with tf.device("/gpu:0"):
        item_flat = tf.layers.flatten(item_max_pool1)
    with tf.device("/gpu:1"):
        item_dense = tf.layers.dense(item_flat, 64, activation=tf.nn.relu)
    
    predictions = tf.reduce_sum( tf.multiply( user_dense, item_dense ), 1, keep_dims=True )
    
    output = {
        "rating": predictions,
        "user_review_embedding": user_flat,
        "item_review_embedding": item_flat
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "mean square error": tf.metrics.mean_squared_error(
            labels=labels, predictions=predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# # Test Forward

# In[3]:


truncate_len = 800
batch_size = 32


# In[4]:


def get_truncate_fn(trunc_len):
    def truncate_fn(user, item, rating):
        return user[:trunc_len], item[:trunc_len], rating
    return truncate_fn


# In[5]:


def split_fn(user, item, rating):
    user = tf.string_split(user)
    item = tf.string_split(item)
    return user.values, item.values, rating


# In[6]:


def parse_fn(record):
    features = {
            "user_review": tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
            "item_review": tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
            "rating": tf.FixedLenFeature([1], tf.float32)
        }
    parsed_features = tf.parse_single_example(record, features)
    return parsed_features["user_review"], parsed_features["item_review"], parsed_features["rating"]


# In[7]:


def get_dataset_iterator(loc, batch_size, max_len, pad_value):
    dataset = tf.data.TFRecordDataset(loc)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.map(parse_fn, num_parallel_calls=batch_size)
    dataset = dataset.map(split_fn, num_parallel_calls=batch_size)
    dataset = dataset.map(get_truncate_fn(max_len), num_parallel_calls=batch_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([max_len], [max_len], [None]), padding_values=(pad_value, pad_value, 0.0))
    dataset = dataset.shuffle(26352, reshuffle_each_iteration=False)
    iterator = dataset.make_one_shot_iterator()
    return iterator


# In[8]:


def train_input_fn():
    train_dataset = get_dataset_iterator(
        loc="data/train.tfrecords",
        batch_size=batch_size,
        max_len=truncate_len,
        pad_value="unk")
    nex = train_dataset.get_next()
    return (nex[0], nex[1]), tf.cast(nex[2], tf.int32)


# In[9]:


def test_input_fn():
    test_dataset = get_dataset_iterator(
        loc="data/test.tfrecords",
        batch_size=batch_size,
        max_len=truncate_len,
        pad_value="unk"
    )
    nex = test_dataset.get_next()
    return (nex[0], nex[1]), nex[2]


# In[10]:


scoring_function = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir="output/model_" + str(int(time.time())))


# In[11]:


# from tensorflow.python import debug as tf_debug
# hook = tf_debug.TensorBoardDebugHook("localhost:6060")
s = time.time()
scoring_function.train(input_fn=train_input_fn) # , hooks=[hook])
e = time.time()
print("ran {} seconds".format(e - s))


# In[ ]:


# scoring_function.evaluate(input_fn=test_input_fn)


# In[ ]:


# action_movie_lover = """i really love action movies they are my favorite kind of movie because i love to watch the good guys
# win. some of my favorite actors are jet li and jackie chan because they were the last great actors who
# actually knew how to fight. modern action movie actors are just pretty faces and the editors swap camera
# angles when supposed hits make contact"""
# action_movie_hater = """I really hate action movies. jackie chan and jet li are the worst. their old fashioned
# special effects, if you can even call them that, are out dated and boring. why are people even still paying
# for them to make movies"""
# item = """this movie is great because of the way the strong, clever, hero saves the day at the end. as always
# jackie chans action directing and stunts are amazing, i'm so glad he doesn't use a stunt double liek
# some other actors that don't need to be mentioned"""


# In[ ]:


# def get_predict_input_fn(user, item, trun_len):
#     def predict_input_fn():
#         original_user_review = [bytes(v, "utf8") for v in user.split()]
#         original_item_review = [bytes(v, "utf8") for v in item.split()]
#         r_user = tf.constant([original_user_review + [b"unk"] * (trun_len - len(original_user_review))])
#         r_item = tf.constant([original_item_review + [b"unk"] * (trun_len - len(original_item_review))])
#         return (r_user, r_item), None
#     return predict_input_fn


# In[ ]:


# prediction = scoring_function.predict(input_fn=get_predict_input_fn(action_movie_lover, item, truncate_len))
# print(next(prediction))


# In[ ]:


# prediction = scoring_function.predict(input_fn=get_predict_input_fn(action_movie_hater, item, truncate_len))
# print(next(prediction))

