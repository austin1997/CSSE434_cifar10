from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
############################################
import os
import random
import gc
from scipy.io import wavfile
from scipy.signal import spectrogram
import wave
import pyaudio
import numpy as np
import pylab
from scipy.fftpack import fft,ifft
import tensorflow as tf
############################################

'''
fs, wave_data = wavfile.read("E:\\青春\\新しいフォルダー\\第一季\\[Kamigami] Yahari Ore no Seishun Love Come wa Machigatteiru 01 [1920x1080 AVC FLAC].wav")
if (os.path.exists('E:\\青春\\新しいフォルダー\\第一季\\01') == False):
	os.makedirs('E:\\青春\\新しいフォルダー\\第一季\\01')

f = open("E:\\青春\\[Kamigami] Yahari Ore no Seishun Love Come wa Machigatteiru [1920x1080 AVC FLAC]\\[Kamigami] Yahari Ore no Seishun Love Come wa Machigatteiru 01 [1920x1080 AVC FLAC].Jap.ass", 'r',encoding = 'utf-8')


wave_data = wave_data.T
wave_data = wave_data[0]
'''
#############################################################
'''
tt = numpy.arange(100000)
pylab.plot(tt, wave_data)
pylab.show()
'''
#############################################################
#返回fs，wave_data
def readaudio(filepath):
	return wavfile.read(filepath)

def writeaudio(filepath, fs, wave_data):
	wavfile.write(filepath, fs, wave_data)

def subs_handler(filepath,fs):
	f = open(filepath, 'r',encoding = 'utf-8')
	lines = f.readlines()
	f.close()
	length = len(lines)
	i = 0
	while lines[i].find('---Sub-JP----') == -1:
		i+=1
	i+=1
	# 找到台词第一句
	dialogue = []
	start_sample = []
	end_sample = []
	num = 0
	print (i)
	while i<length:
#		print (lines[i])
		if (lines[i].find('Sub-Jap') == -1):
			i+=1
			continue
		temp1 = lines[i].split(',,')
		dialogue.append(temp1[2].replace("\n",""))
#		print (dialogue[num]) #台词内容
		temp2 = temp1[0].split(',')
		start_temp = temp2[1]
		end_temp = temp2[2]
		start_min = int(temp2[1].split(':')[1])
		end_min = int(temp2[2].split(':')[1])
		temp2[1] = temp2[1].split(':')[2]
		temp2[2] = temp2[2].split(':')[2]
		start_sec = float(temp2[1])
		end_sec = float(temp2[2])
#		elapse_time.append(end_time - start_time)
#		print (start_min)
#		print (start_sec)
#		print (end_min)
#		print (end_sec)
		start_sample.append(int(start_sec*fs) + start_min *60*fs)
		end_sample.append(int(end_sec*fs) + end_min *60*fs)
#		print (elapse_time[num])
		i+=1
		num+=1
	return dialogue, start_sample, end_sample

def getfilepath(rootfilepath, dialogue, num):
	if (os.path.exists(rootfilepath) == False):
		os.makedirs(rootfilepath)
	return rootfilepath + "\\" + str(num).zfill(4) + " - " + dialogue + ".wav"

def writeaudio(filepath, fs, wave_data):
	wavfile.write(filepath, fs, wave_data)
	
def getspectrum(filepath, label, maxFrequency=5000, mintime=20, tres=100):
	fs, wave_data = wavfile.read(filepath)
	framelength = np.floor((tres/1000.0)*fs)
	numframes = np.floor(len(wave_data)/framelength)
	f, t, Sxx = spectrogram(wave_data, fs=fs, window=np.hanning(framelength), noverlap=0.5*framelength, nfft=framelength)
	if (len(t) < mintime):
		print (len(t))
		print(t)
		print (filepath)
		raise ("文件过短")
	
#	specT = []
	specSxx = []
	artistLabel = []
	f = f[:int(maxFrequency/10)]
#	print (Sxx.shape)
	Sxx = Sxx[:len(f),:]
#	print (Sxx.shape)
	pieceNum = int(len(t) / mintime)
	i = 0
	if label == 0:
		label = [1., 0., 0., 0.]
	elif label == 1:
		label = [0., 1., 0., 0.]
	elif label == 2:
		label = [0., 0., 1., 0.]
	elif label == 3:
		label = [0., 0., 0., 1.]
	while i < pieceNum:
#		time = t[i*mintime:((i+1)*mintime-1)]
#		specT.append(time)
#		print ((Sxx[:,i*mintime:((i+1)*mintime)]).shape) #[500,20]
		specSxx.append(Sxx[:,i*mintime:((i+1)*mintime)])
		artistLabel.append(label)
		i+=1
	if pieceNum * i < len(t):
#		time = t[(-mintime):]
#		specT.append(time)
#		print ((Sxx[:,(-mintime):]).shape)
		specSxx.append(Sxx[:,(-mintime):])
		artistLabel.append(label)

	
	return specSxx, artistLabel	#[-1,500,20]

def getAllSpectrum(rootpath, label, maxFrequency=5000, mintime=28, tres=100):
	list = os.listdir(rootpath)
	f = 0
#	fileSpecT = []
	fileSpecSxx = []
	fileArtistLabel = []
	time = (mintime + 1) * 0.05
	small = 0
	for i in range(len(list)):
		path = os.path.join(rootpath, list[i])
		if os.path.isfile(path):
			wf = wave.open(path, 'rb')
			nframes = wf.getnframes()
			framerate = wf.getframerate()
			if nframes/framerate < time:
				small+=1
				continue
#			print (nframes/framerate)
			specSxx, artistLabel = getspectrum(path, label, maxFrequency, mintime, tres)
#			fileSpecT.append(specT)
			fileSpecSxx.extend(specSxx)
			fileArtistLabel.extend(artistLabel)
	print (small)
	return fileSpecSxx, fileArtistLabel	#[-1,500,20]
	
def getData(filepath, artists):
	list = os.listdir(filepath)
	data = []
	label = []
	for i in range(len(list)):
		path = os.path.join(filepath, list[i])
		if os.path.isdir(path) and (list[i] in artists):
			fileSpecSxx, fileArtistLabel = getAllSpectrum(path, label=artists.index(list[i]))
			data.extend(fileSpecSxx)
			label.extend(fileArtistLabel)
	return data, label	#[-1,500,20]
	
BATCH_SIZE = 50	
	
def main():
	rootfilepath = "E:\\青春\\新しいフォルダー\\第一季\\"
	
	'''
	i = 1
	while i <= 13:
#		print (i)
		audiopath = rootfilepath + "[Kamigami] Yahari Ore no Seishun Love Come wa Machigatteiru " + str(i).zfill(2) + " [1920x1080 AVC FLAC].wav"
		subsfilepath = "E:\\青春\\[Kamigami] Yahari Ore no Seishun Love Come wa Machigatteiru [1920x1080 AVC FLAC]\\[Kamigami] Yahari Ore no Seishun Love Come wa Machigatteiru " + str(i).zfill(2) + " [1920x1080 AVC FLAC].Jap.ass"
		outputfilepath = rootfilepath + str(i).zfill(2)
		fs, wave_data = readaudio(audiopath)
		wave_data = wave_data.T
		wave_data = wave_data[0]
		dialogue, start_sample, end_sample = subs_handler(subsfilepath,fs)
		length = len(dialogue)
#		print (wave_data.shape)
		for j in range(length):
			outfilepath = getfilepath(outputfilepath, dialogue[j].replace("\\",""), j)
			writeaudio(outfilepath, fs, wave_data[start_sample[j]:end_sample[j]])
#			print (j)
		i+=1
	'''
#	音频分割完毕
	dialogue = None
	start_sample = None
	end_sample = None
	gc.collect()
#	开始获取频谱
	trainingrootfilepath = "E:\\青春\\新しいフォルダー\\第一季\\01"
	testingrootfilepath = "E:\\青春\\新しいフォルダー\\第一季\\02"
	artists = ["比企谷八幡", "雪ノ下雪乃", "由比ヶ浜結衣", "平塚静"]
	train_data, train_label = getData(trainingrootfilepath, artists)	#[-1,500,20]
	test_data, test_label = getData(testingrootfilepath, artists)
	'''
	for i in range(len(train_data)):
		train_data[i] = train_data[i].flatten()
	i = 0
	for i in range(len(test_data)):
		test_data[i] = test_data[i].flatten()
	'''	
	print (train_data[0].shape)
	print (len(train_label))
	print (len(test_data))
	print (len(test_label))
	train(train_data, train_label, test_data, test_label)
	
	
'''	
	fileSpecSxx, fileArtistLabel = getAllSpectrum(trainingrootfilepath + "\\比企谷八幡", "比企谷八幡")
	print (len(fileSpecSxx))
	print (len(fileArtistLabel))
'''	
def next_batch(batch_size, input_data, input_label):
		data = []
		label = []
		i = 0
		max = len(input_data) - 1
		for i in range(batch_size):
			index = random.randint(0, max)
			data.append(input_data[index])
			label.append(input_label[index])
		return data, label

def train(train_data, train_label, test_data, test_label):
	x = tf.placeholder(tf.float32, [None, 500, 28])
	y_ = tf.placeholder(tf.float32, [None, 4])
	y_conv, keep_prob = deepnn(x)
	cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(20000):
			data,label = next_batch(BATCH_SIZE, train_data, train_label)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={
									x: data, y_: label, keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))

				print('test accuracy %g' % accuracy.eval(feed_dict={
					x: data, y_: label, keep_prob: 1.0}))
			train_step.run(feed_dict={x: data, y_: label, keep_prob: 0.5})

		print('test accuracy %g' % accuracy.eval(feed_dict={
					x: data, y_: label, keep_prob: 1.0}))
	
	
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
	x_image = tf.reshape(x, [-1, 500, 20, 1])

	# First convolutional layer - maps one grayscale image to 32 feature maps.
	W_conv1 = weight_variable([50, 2, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	# Pooling layer - downsamples by 2X.
	h_pool1 = max_pool_2x2(h_conv1)

	# Second convolutional layer -- maps 32 feature maps to 64.
	W_conv2 = weight_variable([50, 2, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	# Second pooling layer.
	h_pool2 = max_pool_2x2(h_conv2)
	
#	reshape = tf.reshape(h_pool2, [BATCH_SIZE, -1])
#	dim = reshape.get_shape()[1].value
#	print ("dim: ")
#	print (reshape.shape)
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


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
'''
def train(train_data, train_label, test_data, test_label):
	sess = tf.InteractiveSession()
	# Create a multilayer model.
	
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 10000], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 4], name='y-input')
		
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
			
	hidden1 = nn_layer(x, 10000, 2000, 'layer1')
	
	with tf.name_scope('dropout1'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped1 = tf.nn.dropout(hidden1, keep_prob)
		
	hidden2 = nn_layer(dropped1, 2000, 1000, 'layer2')
	
	with tf.name_scope('dropout2'):
		dropped2 = tf.nn.dropout(hidden2, keep_prob)

	hidden3 = nn_layer(dropped2, 1000, 700, 'layer3')
	
	with tf.name_scope('dropout3'):
		dropped3 = tf.nn.dropout(hidden3, keep_prob)
	
	hidden4 = nn_layer(dropped3, 700, 200, 'layer4')
	
	with tf.name_scope('dropout4'):
		dropped4 = tf.nn.dropout(hidden4, keep_prob)
	
	y = nn_layer(dropped4, 200, 4, 'layer5', act=tf.identity)
	
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
	
	# Merge all the summaries and write them out to
	# /tmp/tensorflow/audio/logs/audio_handler (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('/tmp/tensorflow/audio/logs/audio_handler/train', sess.graph)
	test_writer = tf.summary.FileWriter('/tmp/tensorflow/audio/logs/audio_handler/test')
	tf.global_variables_initializer().run()
	
	def next_batch(batch_size):
		data = []
		label = []
		i = 0
		max = len(train_data) - 1
		for i in range(batch_size):
			index = random.randint(0, max)
			data.append(train_data[index])
			label.append(train_label[index])
		return data, label
		
		
	
	# Train the model, and also write summaries.
	# Every 10th step, measure test-set accuracy, and write test summaries
	# All other steps, run train_step on training data, & add training summaries
	
	def feed_dict(train):
		"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
		if train:
			xs, ys = next_batch(100)
			k = 0.9
		else:
			xs, ys = test_data, test_label
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k}
	
	for i in range(1000):
		if i % 10 == 0:  # Record summaries and test-set accuracy
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			print('Accuracy at step %s: %s' % (i, acc))
		else:  # Record train set summaries, and train
			if i % 100 == 99:  # Record execution stats
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step],
										feed_dict=feed_dict(True),
										options=run_options,
										run_metadata=run_metadata)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary, i)
				print('Adding run metadata for', i)
			else:  # Record a summary
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
				train_writer.add_summary(summary, i)
	train_writer.close()
	test_writer.close()
'''
	
if __name__ == "__main__":
    main()

	
'''	
start_nsample = start_time*fs
end_nsample = end_time*fs
print (wave_data[int(start_nsample):int(end_nsample)].shape)
wavfile.write("E:\\青春\\新しいフォルダー\\第一季\\01\\test.wav", fs, wave_data[int(start_nsample):int(end_nsample)])
'''
