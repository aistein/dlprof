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
		with open('timeline_02.json', 'w') as f:
			f.write(chrome_trace)

	print(" took {} seconds".format(time.time() - start_time))
checkMM()
