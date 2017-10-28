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
output_file = "./data/Flicker8k_training_feat.mat"
train_file_names = []
target_height = 224
target_width = 224
state={"key": []}

with open(train_image_filename_path, 'r') as f:
	train_file_names = f.read().splitlines()

counter =0



with tf.device('/cpu:0'):
	sess = tf.Session()
	images = tf.placeholder("float", [1, 224, 224, 3])
	vgg = vgg16.Vgg16("./vgg16.npy")
	vgg.build(images)

	sess.run(tf.global_variables_initializer())

	for filename in train_file_names:
		counter+=1
		print(str(counter) + " / " + str(len(train_file_names)))
		print(filename)
		img = Image.open(train_image_path + filename)
		img = img.resize((224, 224))
		# img.show()
		img = np.array(img)
		img = img.reshape((1, 224, 224, 3))

		
		feed_dict = {images: img}

		
		# with tf.name_scope("content_vgg"):
		# 	vgg.build(images)
		# # print(vgg.data_dict["conv1_1"])

		feat = sess.run(vgg.relu7, feed_dict=feed_dict)
		# state[filename] = feat

with open(output_file, 'wb') as f:
	pickle.dump(state, f)


