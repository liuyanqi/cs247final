import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import vgg16

model_path = './models/'
feature_path = './data/feats.npy'
annotation_path = './data/results_20130124.token'

image_path = "./image/10815824_2997e03d76.jpg"
vocab_dict = dict()
idxtoword = dict()
word_count = dict()
vocabSz = 0
batchSz = 1
featureSz = 4096
hiddenSz = 256
embedSz = 256
index = 6000
word_count_thres = 30
epoch=150

def get_data(annotation_path, feature_path):
     annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
     return np.load(feature_path,'r'), annotations['caption'].values
feat, captions = get_data(annotation_path, feature_path)

print("done reading the files")
print(captions.shape)
print(feat.shape)
trainSz = feat.shape[0]

all_words = []
for line in captions:
	for word in line.lower().split():
		all_words.append(word)
		word_count[word] = word_count.get(word,0) +1
print("load all the words")

vocab_dict["START"] = 0
vocabSz = 1
vocab = [w for w in word_count if word_count[w] >= word_count_thres]
for v in vocab:
	vocab_dict[v] = vocabSz
	idxtoword[vocabSz] = v
	vocabSz +=1

print vocabSz

def read_image(image_path):
	img = Image.open(image_path)
	img = img.resize((224, 224))
	img = np.array(img)
	return img

image = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
vgg = vgg16.Vgg16("./vgg16.npy")
vgg.build(image)

img = tf.placeholder(tf.float32, shape=[batchSz, featureSz])
img_embedding = tf.Variable(tf.random_normal([featureSz, embedSz], stddev=0.1))
img_embedding_bias = tf.Variable(tf.random_normal([embedSz], stddev=0.1))

W = tf.Variable(tf.random_normal([hiddenSz, vocabSz], stddev=0.1))
b = tf.Variable(tf.random_normal([vocabSz], stddev=0.1))

rnn = tf.contrib.rnn.BasicLSTMCell(hiddenSz)

E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
state = rnn.zero_state(batchSz, tf.float32)
maxlen =15

all_words = []
image_embedding = tf.matmul(img, img_embedding) + img_embedding_bias
with tf.variable_scope("RNN"):
	output, state = rnn(image_embedding, state)
	previous_word = tf.nn.embedding_lookup(E, [0])

	for i in range(maxlen):
		tf.get_variable_scope().reuse_variables()

		output, state = rnn(previous_word, state)
		logits = tf.matmul(output, W) + b
		best_word = tf.argmax(logits, 1)


		previous_word = tf.nn.embedding_lookup(E, best_word)
		all_words.append(best_word)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

image_input = read_image(image_path)

feat = sess.run(vgg.relu7, feed_dict={image: [image_input]})

saved_path = tf.train.latest_checkpoint(model_path)
saver.restore(sess, saved_path)

generated_caption_index = sess.run(all_words, feed_dict={img: feat})
generated_caption = [idxtowrod[ind] for ind in generated_caption_index]
print(generated_sentence)





