import numpy as np

dataset = './data/Flicker8k_test/Flickr8k.token.txt'

f8kraw = open(dataset).readlines()
dic = {}
maxlen = 0

#better way?
for line in f8kraw:
	curlen = len(line.split()) - 1
	maxlen = curlen if maxlen < curlen else maxlen

# for the longest string
maxlen = maxlen + 2

for line in f8kraw:
	key = str(line.split()[0][:-2])
	if key not in dic:
		dic[key] = []
	val = f8kraw[0].split()[1:]
	val.insert(0, 'START')
	val.extend(['STOP'] * (maxlen - len(val)))
	dic[key].append(val)




	
