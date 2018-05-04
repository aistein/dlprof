---
exerpt: TFRecrds and the api for creating them are poorly documented in spite of being the preferred method of ingesting data. Here we give you some practical examples of creating and reading tfrecords.
---

Every deep neural network training algorithm revolves around a stream of input data to train on. In Tensorflow the preferred method for data input is via tfrecords. Tfrecords are a binary file format of your data using Google's protocol buffers which provide fast and efficient disk reads. Tfrecords also allow datasets to be split across multiple files in case they do not fit in memory.

The list of advantages is long, but unfortunately in the mass of improvement and development done on Tensorflow, the tfrecords api was left poorly documented and poorly explained. One issue raised on github sums this up quite nicely.

![Documentation Missing]({{ "/dlprof/assets/tfrecords_documentation_dne.png" }})

So, how exactly do we create tfrecords? Below we have provided an example of turning strings into tfrecords of variable length. This example is important because all the examples in Tensorflow documentation transform images of the same size into TFRecrds of the same size. We, on the other hand will be working with text data, and would like tensorflow to handle the embedding creation and word-to-index transformation for us.

```python
1.  def to_bytearray_feature(value):
2.      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, "utf8")]))
3.  def wrap_float_value(value):
4.      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
5.  
6.  tfrecords_filename = "data/demo.tfrecords"
7.  writer = tf.python_io.TFRecordWriter(tfrecords_filename)
8.  
9.  for user, item, rating in zip(train_user_lines, train_item_lines, train_ratings):
10.     example = tf.train.Example(
11.         features=tf.train.Features(
12.             feature={
13.                 'user_review': to_bytearray_feature(user),
14.                 'item_review': to_bytearray_feature(item),
15.                 'rating': wrap_float_value(float(rating))
16.             }
17.         )
18.     )
19. 
20.     writer.write(example.SerializeToString())
```

First, we will analyze the first two lines, inside out.

1. `bytes(value, "utf8")` - This converts the input into it's utf-8 encoded representation
1. `tf.train.BytesList(value=[...])` - This method has one required argument, value, which must by a *list of bytes*. For us this is a list of length one where the bytes represent the original sentence.
1. `tf.train.Feature(bytes_list=...)` - This creates a Tensorflow Feature object, and accepts one of `bytes_list, float_list, int64_list`. In our case we pass `bytes_list`

Lines 3 - 4 are extremely similar except we create a FloatList. Again, this list has exactly one element, the input value. Continuing on, in lines 11 - 17 we create a set of Features. Features take one named input `feature`, which must be a dictionary of string to `tf.train.Feature` objects. Notice that at this point we are using the previously defined functions to create `Feature` objects around the `user, item, rating` objects. In this specific case user and item are both a string, but rating is an integer which is why it needs to be cast to a float. Finally, lines 10 - 20 create an `Example` object, this object takes one `features` input. If you're wondering why this is wrapped so many times, you aren't alone, I wondered the same thing. This api could probably be more well designed by wrapping our dictionary for us, or taking native python dictionary objects representing features directly as input to the `Example` constructor.

In lines 6 - 7 we create a writer object and specify the file it will write to. On line 20 we write our example object to disk, serialized as a string.

It is important to note that in order to read this data, the `user_review` field of each `Features` object must be the same length. This is the reason we created a `BytesList` of length 1 rather than splitting our sentence into multiple words in advance.

Now we will move on to reading our data from the demo.tfrecords file we wrote.

```python
1.  def parse_fn(record):
2.      features = {
3.              "user_review": tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
4.              "item_review": tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
5.              "rating": tf.FixedLenFeature([1], tf.float32)
6.          }
7.      parsed_features = tf.parse_single_example(record, features)
8.      return parsed_features["user_review"], parsed_features["item_review"], parsed_features["rating"]
9.  
10. def split_fn(user, item, rating):
11.     user = tf.string_split(user)
12.     item = tf.string_split(item)
13.     return user.values, item.values, rating
14. 
15. def truncate_fn(user, item, rating):
16.     return user[:400], item[:400], rating
17. 
18. dataset = tf.data.TFRecordDataset("data/demo.tfrecords")
19. dataset = dataset.map(parse_fn)
20. dataset = dataset.map(split_fn)
21. dataset = dataset.map(truncate_fn)
22. dataset = dataset.padded_batch(16, padded_shapes=([400], [400], [None]), padding_values=("unk", "unk", 0.0))
23. iterator = dataset.make_one_shot_iterator()
24. data_point = iterator.get_next()
25. data_point[0].eval(session=tf.Session())
```

This time it is best to start at the bottom. Line 18 creates a TFRecordDataset from our file. In our case there is one file, but if there were multiple files and our data did not fit in memory we could pass a list of files instead. Next on line 19 we call our `parse_fn` on each input value.

`parse_fn` calls `tf.parse_single_example` where each example is expected to have three fields defined as above. Note that we used tf.FixedLenSequenceFeature because we found it the simplest api to work with. For example, `user_review: tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)` means that we expect each `user_review` to have one, undefined dimension and be of type `tf.string`, where missing values are also allowed. Note that when using the `tf.string` type, `allow_missing` was required. 

Next we call `split_fn` to split our Tensorflow strings into arrays of strings. Since `tf.string_split` returns a map of indices and values, we must return only the values from our split.

Next we call the `truncate_fn` to shorten any strings longer than our predefined length to the appropriate size. For us, this was required because we eventually feed this data into a convolutional network, which takes a fixed length input. 

Finally, we create `padded_batch`es. The first input is the batch size, 16, then we have the shape of each element output by the previous map function, and lastly the values to pad with. Note that we had to provide a pad value for the rating even though we know it will never need to be padded.

Lastly, we call `make_one_shot_iterator`, get a value, and evaluate it to confirm we can read from our dataset!

The TFRecords api is considered the standard data input format for Tensorflow models and based on Google's protobufs. TFRecords provide many advantages like speed and an easy way to read data too large to fit in memory. Unfortunately the api is extremely difficult to use, and we find ourselves wrapping objects in objects where sane defaults could be much easier to work with.

