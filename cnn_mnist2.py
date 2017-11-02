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

  IMAGE_PIXELS=28

  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)

  # Parameters
  hidden_units = 128
  batch_size   = args.batch_size

  # Get TF cluster and server instances
  cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)
  

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

      print("In a TFCluster.")

      # Input placeholders
      with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input') 
      
      with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)	

      # We can't initialize these variables to 0 - the network will get stuck.
      def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

      def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

      def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)	

      def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
          # This Variable will hold the state of the weights for the layer
          with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
          with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
          with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
          activations = act(preactivate, name='activation')
          tf.summary.histogram('activations', activations)
          return activations

      hidden1 = nn_layer(x, 784, 500, 'layer1')
	  
      with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

      # Do not apply softmax activation yet, see below.
      y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

      with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
          cross_entropy = tf.reduce_mean(diff)

      tf.summary.scalar('cross_entropy', cross_entropy)
	  
      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(
                cross_entropy)
	  
      with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('accuracy', accuracy)
      label = tf.argmax(y_, 1, name="label")
      prediction = tf.argmax(y, 1,name="prediction")  

      # Merge all the summaries and write them out to
      # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
      merged = tf.summary.merge_all()
	  
	  


      

      global_step = tf.Variable(0)






#      saver = tf.train.Saver()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
#    logdir = TFNode.hdfs_path(ctx, args.model)
    logdir = "/tmp/" + args.model
    print("tensorflow model path: {0}".format(logdir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" %(worker_num), graph=tf.get_default_graph())

    if args.mode == "train":
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               init_op=init_op,
                               summary_op=None,
#                               summary_writer=summary_writer,
                               global_step=global_step,
                               stop_grace_secs=300,
#                               save_model_secs=10
                               )
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
        step = step + 1
        # using feed_dict
        batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
        feed = {x: batch_xs, y_: batch_ys, keep_prob: 0.9}

        if len(batch_xs) > 0:
          if args.mode == "train":
            summary, _, _ = sess.run([merged, train_step, global_step], feed_dict=feed)
            # print accuracy and save model checkpoint to HDFS every 100 steps
            if (step % 100 == 0):
              labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
              for l,p in zip(labels,preds):
			    print("{0} step: {1} accuracy: {2}, Label: {3}, Prediction: {4}".format(datetime.now().isoformat(), step, acc, l, p))
              
#              results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)]
#              tf_feed.batch_results(results)

            if sv.is_chief:
              summary_writer.add_summary(summary, step)
          else: # args.mode == "inference"
            labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict=feed)

            results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)]
            tf_feed.batch_results(results)
            print("acc: {0}".format(acc))

      if sv.should_stop() or step >= args.steps:
        tf_feed.terminate()

    # Ask for all the services to stop.
    print("{0} stopping supervisor".format(datetime.now().isoformat()))
    sv.stop()