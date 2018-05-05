import tensorflow as tf
import time
tf.set_random_seed(42)
A = tf.random_normal([30000,30000])
B = tf.random_normal([30000,30000])
def checkMM():
	start_time = time.time()
	with tf.Session() as sess:
		print( sess.run( tf.reduce_sum( tf.matmul(A,B) ) ) )
	print(" took {} seconds".format(time.time() - start_time))
checkMM()
