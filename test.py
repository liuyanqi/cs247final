import numpy as np
import os
import pickle
import pandas as pd

train_image_filename_path = "./data/Flicker8k_text/Flickr_8k.trainImages.txt"
annotation_set = './data/Flicker8k_text/Flickr8k.token.txt'
feature_input = "./data/Flicker8k_training_feat.mat"
feature_file = "./data/Flicker8k_training_feat1.mat"
caption_file = "./data/Flicker8k_training_text.mat"
dataset = './data/Flicker8k_text/Flickr8k.token.txt'
train_file_names = []

with open(train_image_filename_path, 'r') as f:
	train_file_names = f.read().splitlines()

with open(feature_input, 'r') as f:
	feature_state = pickle.load(f)

caption = {}
captions = open(dataset).readlines()

for line in captions:
	filename = str(line.split()[0][:-2])
	if filename not in caption:
		caption[filename] = []
	title = line.split('\t')[1:]
	caption[filename].append(title)

total_feature= []
total_caption = []
for key, val in feature_state.items():
	if key in caption:
		val = np.array(val)
		val = val.reshape(4096,)
		titles = caption[key]
		for title in titles:
			total_feature.append(val)
			total_caption.append(title[0])

print(np.array(total_feature).shape)
print(np.array(total_caption).shape)

with open(feature_file, "wb") as f:
	pickle.dump(np.array(total_feature), f)
with open(caption_file, "wb") as f:
	for title in captions:
		f.write(title)
	f.close()




