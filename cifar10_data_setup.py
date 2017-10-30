# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import os
import tensorflow as tf
from array import array
#from tensorflow.contrib.learn.python.learn.datasets import mnist
import cifar10_2
import cPickle

def toTFExample(image, label):
	"""Serializes an image/label as a TFExample byte string"""
	example = tf.train.Example(
		features = tf.train.Features(
			feature = {
				'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.astype("int64"))),
				'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image.astype("int64")))
			}	
		)
	)
	return example.SerializeToString()

def fromTFExample(bytestr):
	"""Deserializes a TFExample from a byte string"""
	example = tf.train.Example()
	example.ParseFromString(bytestr)
	return example

def toCSV(vec):
	"""Converts a vector/array into a CSV string"""
	return ','.join([str(i) for i in vec])

def fromCSV(s):
	"""Converts a CSV string to a vector/array"""
	return [float(x) for x in s.split(',') if len(s) > 0]
	
def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def writeMNIST(sc, input_images, output, format, num_partitions):
	"""Writes MNIST image/label vectors into parallelized files on HDFS"""
	# load MNIST gzip into memory
#	with open(input_images, 'rb') as f:
#	images, labels = cifar10_2.distorted_inputs(input_images)
	dict1 = unpickle(os.path.join(input_images, 'cifar-10-batches-py', 'data_batch_1'))
	dict2 = unpickle(os.path.join(input_images, 'cifar-10-batches-py', 'data_batch_2'))
	dict3 = unpickle(os.path.join(input_images, 'cifar-10-batches-py', 'data_batch_3'))
	dict4 = unpickle(os.path.join(input_images, 'cifar-10-batches-py', 'data_batch_4'))
	dict5 = unpickle(os.path.join(input_images, 'cifar-10-batches-py', 'data_batch_5'))
	dict6 = unpickle(os.path.join(input_images, 'cifar-10-batches-py', 'test_batch'))
#	dictMerged = dict1.copy()
#	dictMerged.update(dict2)
#	dictMerged.update(dict3)
#	dictMerged.update(dict4)
#	dictMerged.update(dict5)
	
	images1 = dict1['data']
	labels1 = numpy.array(dict1['labels'])
	images2 = dict2['data']
	labels2 = numpy.array(dict2['labels'])
	images3 = dict3['data']
	labels3 = numpy.array(dict3['labels'])
	images4 = dict4['data']
	labels4 = numpy.array(dict4['labels'])
	images5 = dict5['data']
	labels5 = numpy.array(dict5['labels'])
	images6 = dict6['data']
	labels6 = numpy.array(dict6['labels'])
	
	mergedImage = numpy.vstack((images1, images2))
	mergedImage = numpy.vstack((mergedImage, images3))
	mergedImage = numpy.vstack((mergedImage, images4))
	mergedImage = numpy.vstack((mergedImage, images5))
	mergedLabel = numpy.vstack((labels1, labels2))
	mergedLabel = numpy.vstack((mergedLabel, labels3))
	mergedLabel = numpy.vstack((mergedLabel, labels4))
	mergedLabel = numpy.vstack((mergedLabel, labels5))

	shape = mergedImage.shape
	b = numpy.zeros((shape[0], 10))
	c = numpy.zeros((10000, 10))
	b[numpy.arange(shape[0]), mergedLabel] = 1
	c[numpy.arange(10000), labels6] = 1
	
	print("images.shape: {0}".format(shape))          # 60000 x 28 x 28
	print("labels.shape: {0}".format(b.shape))   # 60000 x 10

	
	# create RDDs of vectors
	imageRDD = sc.parallelize(mergedImage.reshape(shape[0], 32, 32, 3), num_partitions)
	labelRDD = sc.parallelize(b, num_partitions)

	output_train_images = args.output + "/train" + "/images"
	output_train_labels = args.output + "/train" + "/labels"
	
	# create RDDs of vectors
	imageRDD = sc.parallelize(images6.reshape(-1, 32, 32, 3), num_partitions)
	labelRDD = sc.parallelize(c, num_partitions)

	output_test_images = args.output + "/test" + "/images"
	output_test_labels = args.output + "/test" + "/labels"

	# save RDDs as specific format
	if format == "pickle":
		imageRDD.saveAsPickleFile(output_images)
		labelRDD.saveAsPickleFile(output_labels)
	elif format == "csv":
		imageRDD.map(toCSV).saveAsTextFile(output_train_images)
		labelRDD.map(toCSV).saveAsTextFile(output_train_labels)
		imageRDD.map(toCSV).saveAsTextFile(output_test_images)
		labelRDD.map(toCSV).saveAsTextFile(output_test_labels)
	elif format == "csv2":
		imageRDD.map(toCSV).zip(labelRDD).map(lambda x: str(x[1]) + "|" + x[0]).saveAsTextFile(output)
	else: # format == "tfr":
		tfRDD = imageRDD.zip(labelRDD).map(lambda x: (bytearray(toTFExample(x[0], x[1])), None))
		# requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
		tfRDD.saveAsNewAPIHadoopFile(output, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
								keyClass="org.apache.hadoop.io.BytesWritable",
								valueClass="org.apache.hadoop.io.NullWritable")
#  Note: this creates TFRecord files w/o requiring a custom Input/Output format
#  else: # format == "tfr":
#    def writeTFRecords(index, iter):
#      output_path = "{0}/part-{1:05d}".format(output, index)
#      writer = tf.python_io.TFRecordWriter(output_path)
#      for example in iter:
#        writer.write(example)
#      return [output_path]
#    tfRDD = imageRDD.zip(labelRDD).map(lambda x: toTFExample(x[0], x[1]))
#    tfRDD.mapPartitionsWithIndex(writeTFRecords).collect()

def readMNIST(sc, output, format):
	"""Reads/verifies previously created output"""

	output_images = output + "/images"
	output_labels = output + "/labels"
	imageRDD = None
	labelRDD = None

	if format == "pickle":
		imageRDD = sc.pickleFile(output_images)
		labelRDD = sc.pickleFile(output_labels)
	elif format == "csv":
		imageRDD = sc.textFile(output_images).map(fromCSV)
		labelRDD = sc.textFile(output_labels).map(fromCSV)
	else: # format.startswith("tf"):
		# requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
		tfRDD = sc.newAPIHadoopFile(output, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
							keyClass="org.apache.hadoop.io.BytesWritable",
							valueClass="org.apache.hadoop.io.NullWritable")
		imageRDD = tfRDD.map(lambda x: fromTFExample(str(x[0])))

	num_images = imageRDD.count()
	num_labels = labelRDD.count() if labelRDD is not None else num_images
	samples = imageRDD.take(10)
	print("num_images: ", num_images)
	print("num_labels: ", num_labels)
	print("samples: ", samples)

if __name__ == "__main__":
	import argparse

	from pyspark.context import SparkContext
	from pyspark.conf import SparkConf

	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--format", help="output format", choices=["csv","csv2","pickle","tf","tfr"], default="csv")
	parser.add_argument("-n", "--num-partitions", help="Number of output partitions", type=int, default=10)
	parser.add_argument("-o", "--output", help="HDFS directory to save examples in parallelized format", default="mnist_data")
	parser.add_argument("-r", "--read", help="read previously saved examples", action="store_true")
	parser.add_argument("-v", "--verify", help="verify saved examples after writing", action="store_true")

	args = parser.parse_args()
	print("args:",args)

	sc = SparkContext(conf=SparkConf().setAppName("mnist_parallelize"))

	if not args.read:
		# Note: these files are inside the mnist.zip file
		writeMNIST(sc, "dataset/", args.output + "/train", args.format, args.num_partitions)
		writeMNIST(sc, "dataset/", args.output + "/test", args.format, args.num_partitions)

	if args.read or args.verify:
		readMNIST(sc, args.output + "/train", args.format)