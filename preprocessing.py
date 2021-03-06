import os

import vgg16
import utils

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle



train_image_path = "./data/Flicker8k_Dataset/"
train_image_filename_path = "./data/Flicker8k_text/Flickr_8k.trainImages.txt"
output_file = "./data/Flicker8k_training_conv_feat_1.mat"
train_file_names = []
target_height = 224
target_width = 224
state={"key": []}

with open(train_image_filename_path, 'r') as f:
	train_file_names = f.read().splitlines()

counter =0
batchSz = 20
def dataLoader(train_file_names):
	global counter
	image_batch = []
	print("from: "+ str(counter) +" to: " + str(counter+batchSz))
	for filename in train_file_names[counter: counter+batchSz]:
		img = Image.open(train_image_path + filename)
		img = img.resize((224,224))
		img = np.array(img)
		image_batch.append(img)
	return np.array(image_batch), train_file_names[counter: counter+batchSz]
	

with tf.device('/gpu:0'):
	sess = tf.Session()
	images = tf.placeholder("float", [batchSz, 224, 224, 3])
	vgg = vgg16.Vgg16("./vgg16.npy")
	vgg.build(images)

	sess.run(tf.global_variables_initializer())

	# for filename in train_file_names:
	for counter in range(len(train_file_names)/3, 2*len(train_file_names)/3, batchSz):
		print(str(counter) + " / " + str(len(train_file_names)))
		
		# img = Image.open(train_image_path + train_file_names[counter])
		# img = img.resize((224, 224))
		# # img.show()
		# img = np.array(img)
		# img = img.reshape((1, 224, 224, 3))
		
		
		img, filename_list = dataLoader(train_file_names)

		feed_dict = {images: img}

		
		# with tf.name_scope("content_vgg"):
		# 	vgg.build(images)
		# # print(vgg.data_dict["conv1_1"])

		# feat = sess.run(vgg.relu7, feed_dict=feed_dict)
		tf.get_variable_scope().reuse_variables()

		feat = sess.run(tf.contrib.layers.flatten(vgg.conv5_3), feed_dict=feed_dict)
		print(feat.shape)
		for idx, filename in enumerate(filename_list):
			state[filename] = feat[idx]

with open(output_file, 'wb') as f:
	pickle.dump(state, f)


