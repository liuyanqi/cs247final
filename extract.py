import numpy as np

f8kraw = open("Flickr8k.token.txt").readlines()
dic = {}


# val = f8kraw[0].split()[1:]
# val.insert(0, 'START')
# val.append('STOP')



for line in f8kraw:
	key = str(line.split()[0][:-2])
	if key not in dic:
		dic[key] = []
	val = f8kraw[0].split()[1:]
	val.insert(0, 'START')
	val.append('STOP')
	dic[key].append(val)

print dic
	
