import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import vgg16
from tf_beam_decoder import beam_decoder

model_path = './models3/'
feature_path = './data/feats.npy'
annotation_path = './data/results_20130124.token'

image_path = "./image/44856031_0d82c2c7d1.jpg"
# image_path = "./image/10815824_2997e03d76.jpg"
# image_path = "./image/47871819_db55ac4699.jpg"
# image_path = "./image/women_horse.jpg"
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
     return annotations['caption'].values
captions = get_data(annotation_path, feature_path)

print("done reading the files")
print(captions.shape)



all_words = []
for line in captions:
	for word in line.lower().split():
		all_words.append(word)
		word_count[word] = word_count.get(word,0) +1
print("load all the words")

vocab_dict["START"] = 0
vocabSz = 1
idxtoword[0] = '.'
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
maxlen =20

all_words = []
all_score = []

def outputs_to_score_fn(model_output):
	logits = tf.tensordot(model_output, W, axes=[[2],[0]]) + b
	return tf.nn.log_softmax(logits)

beamSz = 1


image_embedding = tf.matmul(img, img_embedding) + img_embedding_bias
with tf.variable_scope("RNN"):
	output, state = rnn(image_embedding, state)
	previous_word = tf.nn.embedding_lookup(E, [0])

	with tf.variable_scope("RNN_beam") as scope:
		best_sparse, best_logprobs = beam_decoder(
			cell=rnn,
			beam_size=beamSz,
			stop_token= 0,
			initial_state=state,
			initial_input=previous_word,
			tokens_to_inputs_fn= lambda x: tf.nn.embedding_lookup(E, x),
			outputs_to_score_fn = lambda x: outputs_to_score_fn(x),
			max_len=maxlen,
			cell_transform='default',
			output_dense=True,
			scope=scope
			)

	# for i in range(maxlen):
	# 	tf.get_variable_scope().reuse_variables()


	# 	output, state = rnn(previous_word, state)
	# 	prob = tf.matmul(output, W) + b
	# 	best_word = tf.argmax(prob, 1)
	# 	previous_word = tf.nn.embedding_lookup(E, best_word)
	# 	all_words.append(best_word)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

image_input = read_image(image_path)

feat = sess.run(vgg.relu7, feed_dict={image: [image_input]})

saved_path = tf.train.latest_checkpoint(model_path)
saver.restore(sess, saved_path)

bs, bl = sess.run([best_sparse, best_logprobs], feed_dict={img:feat})

# generated_caption_index = sess.run(all_words, feed_dict={img: feat})
generated_caption = [idxtoword[ind] for ind in bs[0]]
# generated_caption = [idxtoword[ind[0]] for ind in generated_caption_index]
# generated_caption = [idxtoword[ind] for ind in bs[0]]
ind_end = np.argmax(np.array(generated_caption)=='.')
generated_caption = generated_caption[:ind_end]

print(generated_caption)





