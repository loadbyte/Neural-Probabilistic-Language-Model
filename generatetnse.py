#Latent Semantic Analysis
import collections
import math
import os
import random
from collections import defaultdict, Counter
from itertools import tee
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np

#context window size
window_size=5

#for Considering only top 10K most frequent words
vocab_size = 10001
#dict for mapping key to list
d = defaultdict(Counter)


"""
takes the input corpus and only takes the top (vocabulary_size)
most frequent words and replaces others with UNK
"""
with open('text8', 'r') as f:
	str = f.read()
words = str.split()

dic_wrd = dict()
data = list()
#count = [['UNK', -1]]
count = []
frq_word = Counter(words)
frq_word = frq_word.most_common(vocab_size - 1)
count.extend(frq_word)

for word, _ in count:
    dic_wrd[word] = len(dic_wrd)

for word in words:
	idx = 0
	if word in dic_wrd :
		idx = dic_wrd[word]
	data.append(idx)
	
reverse_dic_wrd = dict(zip(dic_wrd.values(), dic_wrd.keys()))

print "build dataset"
del words # Hint to reduce memory.
#tlst = data
tlst = [reverse_dic_wrd[i] for i in data]
print "reverse dict"

U = np.load('wrd_embeds.npy')
for val in U[:10,:50]:
	print val

tsne = TSNE(n_components=2, random_state=0)
outputMatrix = tsne.fit_transform(U[0:100, :50])
labels = [reverse_dic_wrd[i] for i in xrange(100)]
#plot_with_labels(outputMatrix, labels)
print "labels Created"

assert outputMatrix.shape[0] >= len(labels), "More labels than embeddings"
plt.figure(figsize=(18, 18))  # in inches
for i, label in enumerate(labels):
	x, y = outputMatrix[i, :]
	plt.scatter(x, y)
	plt.annotate(label, xy=(x, y), xytext=(5, 2),
		textcoords='offset points', ha='right', va='bottom')

plt.savefig('output.png')