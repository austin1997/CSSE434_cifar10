# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

def print_log(worker_num, arg):
	print("{0}: {1}".format(worker_num, arg))

def map_fun(args, ctx):
	from tensorflowonspark import TFNode
	from datetime import datetime
	import math
	import numpy
	import tensorflow as tf
	import time

	worker_num = ctx.worker_num
	job_name = ctx.job_name
	task_index = ctx.task_index
	cluster_spec = ctx.cluster_spec

	IMAGE_PIXELS=32
	def deepnn(x):
		"""deepnn builds the graph for a deep net for classifying digits.

		Args:
			x: an input tensor with the dimensions (N_examples, 784), where 784 is the
			number of pixels in a standard MNIST image.

		Returns:
			A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
			equal to the logits of classifying the digit into one of 10 classes (the
			digits 0-9). keep_prob is a scalar placeholder for the probability of
			dropout.
		"""
		# Reshape to use within a convolutional neural net.
		# Last dimension is for "features" - there is only one here, since images are
		# grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
		x_image = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 3])

		# First convolutional layer - maps one grayscale image to 32 feature maps.
		W_conv1 = weight_variable([5, 5, 3, 64])
		b_conv1 = bias_variable([64])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

		# Pooling layer - downsamples by 2X.
		h_pool1 = max_pool_3x3(h_conv1)

		# Second convolutional layer -- maps 32 feature maps to 64.
		W_conv2 = weight_variable([5, 5, 64, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

		# Second pooling layer.
		h_pool2 = max_pool_3x3(h_conv2)
		
	#	reshape = tf.reshape(h_pool2, [BATCH_SIZE, -1])
	#	dim = reshape.get_shape()[1].value
	#	print ("dim: ")
	#	print (reshape.shape)
	
	reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
		dim = 125*7*64
		
		# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
		# is down to 7x7x64 feature maps -- maps this to 1024 features.
		W_fc1 = weight_variable([dim, 1024])
		b_fc1 = bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, dim])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Dropout - controls the complexity of the model, prevents co-adaptation of
		# features.
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# Map the 1024 features to 10 classes, one for each digit
		W_fc2 = weight_variable([1024, 4])
		b_fc2 = bias_variable([4])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		return y_conv, keep_prob

	def conv2d(x, W):
		"""conv2d returns a 2d convolution layer with full stride."""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


	def max_pool_3x3(x):
		"""max_pool_3x3 downsamples a feature map by 2X."""
		return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
								strides=[1, 2, 2, 1], padding='SAME')


	def weight_variable(shape):
		"""weight_variable generates a weight variable of a given shape."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)


	def bias_variable(shape):
		"""bias_variable generates a bias variable of a given shape."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	# Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
	if job_name == "ps":
		time.sleep((worker_num + 1) * 5)

	# Parameters
	hidden_units = 128
	batch_size   = args.batch_size

	# Get TF cluster and server instances
	cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)
	
	def writeFileToHDFS():
		rootdir = '/tmp/mnist_model'
		client = HdfsClient(hosts='localhost:50070')
		client.mkdirs('/user/root/mnist_model')
		for parent,dirnames,filenames in os.walk(rootdir):
			for dirname in  dirnames:
				print("parent is:{0}".format(parent))
		for filename in filenames:
			client.copy_from_local(os.path.join(parent,filename), os.path.join('/user/root/mnist_model',filename), overwrite=True)


	def feed_dict(batch):
		# Convert from [(images, labels)] to two numpy arrays of the proper type
		images = []
		labels = []
		for item in batch:
			images.append(item[0])
			labels.append(item[1])
		xs = numpy.array(images)
		xs = xs.astype(numpy.float32)
		xs = xs/255.0
		ys = numpy.array(labels)
		ys = ys.astype(numpy.uint8)
		return (xs, ys)

	if job_name == "ps":
		server.join()
	elif job_name == "worker":

		# Assigns ops to the local worker by default.
		with tf.device(tf.train.replica_device_setter(
			worker_device="/job:worker/task:%d" % task_index,
			cluster=cluster)):

			# Placeholders or QueueRunner/Readers for input data
			x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")
			y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

			y_conv, keep_prob = deepnn(x)
			
			cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
			train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
			correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			# Test trained model
#			label = tf.argmax(y_, 1, name="label")
#			prediction = tf.argmax(y, 1,name="prediction")
#			correct_prediction = tf.equal(prediction, label)
#
#			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
#			tf.summary.scalar("acc", accuracy)
#
#			saver = tf.train.Saver()
#			summary_op = tf.summary.merge_all()
			init_op = tf.global_variables_initializer()

		# Create a "supervisor", which oversees the training process and stores model state into HDFS
#		logdir = TFNode.hdfs_path(ctx, args.model)
		logdir = "hdfs:///tmp/" + args.model
		print("tensorflow model path: {0}".format(logdir))
#		summary_writer = tf.summary.FileWriter("tensorboard_%d" %(worker_num), graph=tf.get_default_graph())

		if args.mode == "train":
			sv = tf.train.Supervisor(is_chief=(task_index == 0),
								logdir=logdir,
								init_op=init_op,
								summary_op=None,
								saver=None,
								global_step=global_step,
								summary_writer=None,
								stop_grace_secs=300,
								save_model_secs=10)
		else:
			sv = tf.train.Supervisor(is_chief=(task_index == 0),
								logdir=logdir,
								summary_op=None,
								saver=saver,
								global_step=global_step,
								stop_grace_secs=300,
								save_model_secs=0)

		# The supervisor takes care of session initialization, restoring from
		# a checkpoint, and closing when done or an error occurs.
		with sv.managed_session(server.target) as sess:
			print("{0} session ready".format(datetime.now().isoformat()))

			# Loop until the supervisor shuts down or 1000000 steps have completed.
			step = -1
			tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
			while not sv.should_stop() and not tf_feed.should_stop() and step < args.steps:
				# Run a training step asynchronously.
				# See `tf.train.SyncReplicasOptimizer` for additional details on how to
				# perform *synchronous* training.

				# using feed_dict
				batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
				feed = {x: batch_xs, y_: batch_ys}

				if len(batch_xs) > 0:
					if args.mode == "train":
						_, summary, step = sess.run([train_op, summary_op, global_step], feed_dict=feed)
						# print accuracy and save model checkpoint to HDFS every 100 steps
						if (step % 100 == 0):
							print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step, sess.run(accuracy,{x: batch_xs, y_: batch_ys})))

						if sv.is_chief:
							summary_writer.add_summary(summary, step)
							
					else: # args.mode == "inference"
						labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict=feed)

						results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)]
						tf_feed.batch_results(results)
						print("acc: {0}".format(acc))

			if sv.should_stop() or step >= args.steps:
				tf_feed.terminate()
				writeFileToHDFS()

		# Ask for all the services to stop.
		print("{0} stopping supervisor".format(datetime.now().isoformat()))
		sv.stop()