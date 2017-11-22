import numpy as np
import pickle
import tensorflow as tf

feature_file = "./data/Flicker8k_training_feat1.mat"
caption_file = "./data/Flicker8k_training_text.mat"

vocab_dict = dict()
idxtoword = dict()
vocabSz = 0
batchSz = 20
featureSz = 4096
hiddenSz = 100
embedSz = 256
index = 6000

with open(feature_file, 'r') as f:
	feat = pickle.load(f)
with open(caption_file, 'r') as f:
	captions = pickle.load(f)
print("done reading the files")
all_words = []

for line in captions:
	for word in line.lower().split():
		all_words.append(word)
print("load all the words")
vocab_dict["START"] = 0
vocabSz = 1
vocab = set(all_words)
for v in vocab:
	vocab_dict[v] = vocabSz
	idxtoword[vocabSz] = v
	vocabSz +=1

caption_encode = []
caption_length = []
for line in captions:
	line = line.lower().split()
	line = [vocab_dict[w] for w in line]
	caption_encode.append(line)
	caption_length.append(len(line))
caption_encode = np.array(caption_encode)
maxlen = np.max(caption_length)

#def build model

img = tf.placeholder(tf.float32, shape=[batchSz, featureSz])
caption = tf.placeholder(tf.int32, shape=[batchSz, maxlen])
mask = tf.placeholder(tf.int32, shape=[batchSz, maxlen])

#embed image feature into word embeeding space
img_embedding = tf.Variable(tf.random_normal([featureSz, embedSz], stddev=0.1))
img_embedding_bias = tf.Variable(tf.random_normal([embedSz], stddev=0.1))

#covert LSTM output to word 
W = tf.Variable(tf.random_normal([hiddenSz, vocabSz], stddev=0.1))
b = tf.Variable(tf.random_normal([vocabSz], stddev=0.1))

# getting an initial LSTM embedding from our image_imbedding
rnn = tf.contrib.rnn.BasicLSTMCell(hiddenSz)

E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))

total_loss = 0
with tf.variable_scope("RNN"):
	state = rnn.zero_state(batchSz, tf.float32)
	for i in range(maxlen):
		if i > 0:
			current_embeddings = tf.nn.embedding_lookup(E, caption[:,i-1])
		else:
			current_embeddings = tf.matmul(img, img_embedding) + img_embedding_bias

		if i > 0:
			tf.get_variable_scope().reuse_variables()

		output, state = rnn(current_embeddings, state)
		# if i > 0 :
		# 	print(output)
		# 	total_output.append(list(output))

	#shape of output is batchSz* hiddenSz
	#shape of logits is batchSz* vocabS
		labels = tf.expand_dims(caption[:, i], 1)
		ix_range=tf.range(0, batchSz, 1)
		ixs = tf.expand_dims(ix_range, 1)
		concat = tf.concat([ixs, labels],1)
		onehot = tf.sparse_to_dense(concat, tf.stack([batchSz, vocabSz]), 1.0, 0.0)
		logit = tf.matmul(output, W) + b
		xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)

	 	loss = tf.reduce_sum(xentropy)
	 	total_loss += loss

learning_rate = 0.001

train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())




