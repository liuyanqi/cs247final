import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

feature_file = "./data/Flicker8k_training_feat1.mat"
caption_file = "./data/Flicker8k_training_text.mat"

feature_path = './data/feats.npy'
annotation_path = './data/results_20130124.token'

vocab_dict = dict()
idxtoword = dict()
word_count = dict()
vocabSz = 0
batchSz = 128
featureSz = 4096
hiddenSz = 256
embedSz = 256
index = 6000
word_count_thres = 30

def get_data(annotation_path, feature_path):
     annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
     return np.load(feature_path,'r'), annotations['caption'].values
feat, captions = get_data(annotation_path, feature_path)

# with open(feature_file, 'r') as f:
# 	feat = pickle.load(f)
# with open(caption_file, 'r') as f:
# 	captions = f.readlines()
# captions = np.array(captions)

print("done reading the files")
print(captions.shape)
print(feat.shape)
trainSz = feat.shape[0]

all_words = []
maxlen = 0
for line in captions:
	for word in line.lower().split():
		all_words.append(word)
		word_count[word] = word_count.get(word,0) +1
	if(len(line.split()) > maxlen):
		maxlen = len(line.split())

print("load all the words")
print("maxlen: ", str(maxlen))

vocab_dict["START"] = 0

vocabSz = 1
vocab = [w for w in word_count if word_count[w] >= word_count_thres]
for v in vocab:
	vocab_dict[v] = vocabSz
	idxtoword[vocabSz] = v
	vocabSz +=1


print vocabSz

caption_encode = []
caption_length =[]
for line in captions:
	line = [vocab_dict[w] for w in line.lower().split()[:-1] if w in vocab_dict]
	caption_length.append(len(line))
	while(len(line) < maxlen):
		line.append(0)
	caption_encode.append(line)

caption_encode = np.array(caption_encode)
padd = np.full((len(caption_encode),), 0)
caption_encode = np.column_stack((padd, caption_encode[:,:-1]))

for i in range(5):
	print(caption_encode[i])

print("shape of caption_encoder: ", str(caption_encode.shape))
#def build model

img = tf.placeholder(tf.float32, shape=[batchSz, featureSz])
caption = tf.placeholder(tf.int32, shape=[batchSz, maxlen])
sequence = tf.placeholder(tf.float32, shape=[batchSz])
mask = tf.sequence_mask(sequence, maxlen, tf.float32)

#embed image feature into word embeeding space
img_embedding = tf.Variable(tf.random_normal([featureSz, embedSz], stddev=0.1))
img_embedding_bias = tf.Variable(tf.random_normal([embedSz], stddev=0.1))

#covert LSTM output to word 
W = tf.Variable(tf.random_normal([hiddenSz, vocabSz], stddev=0.1))
b = tf.Variable(tf.random_normal([vocabSz], stddev=0.1))

# getting an initial LSTM embedding from our image_imbedding
rnn = tf.contrib.rnn.BasicLSTMCell(hiddenSz)

E = tf.Variable(tf.random_normal([vocabSz, embedSz], stddev=0.1))
state = rnn.zero_state(batchSz, tf.float32)

total_loss = 0
with tf.variable_scope("RNN"):
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
		if i > 0:
			labels = tf.expand_dims(caption[:, i], 1)
			ix_range=tf.range(0, batchSz, 1)
			ixs = tf.expand_dims(ix_range, 1)
			concat = tf.concat([ixs, labels],1)
			onehot = tf.sparse_to_dense(concat, tf.stack([batchSz, vocabSz]), 1.0, 0.0)
			logit = tf.matmul(output, W) + b
			xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
			xentropy = xentropy * mask[:,i]
		 	loss = tf.reduce_sum(xentropy)
		 	total_loss += loss

learning_rate = 0.001

train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train(feat, captions):
	for ind in range(0, feat.shape[0]-batchSz, batchSz):
		print(str(ind) + " / " + str(feat.shape[0]))
		current_feats = feat[ind: ind+batchSz]
		current_caption = caption_encode[ind: ind+batchSz]
		seq = caption_length[ind: ind+batchSz]


		feedDict = {img: current_feats, caption: current_caption, sequence: seq}
		sessArgs = [total_loss, train_op]
		loss, _ = sess.run(sessArgs, feedDict)
		print(loss)


train(feat, caption_encode)


