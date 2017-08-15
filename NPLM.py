"""
Neural Probabilistic Language Model
Author: Vinod Kumar M
"""
import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter

#hyper parameters
dir_corpus = 'data'
sizeof_batch = 120
window_size = 5
hidden_l_sz = 100
word_dim = 50
num_epochs = 1
grad_clip = 10
vocab_size = 10000
alpha = 0.01
file_name ="text8" #input file name


class Corpusprocess():
    def __init__(self, data_dir, batch_size, seq_length, file_name, vocab_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.context_window_size = seq_length 
        self.batch_ptr = 0
        self.vocab_size = vocab_size
        input_file = os.path.join(data_dir, file_name) 
        self.preprocess(input_file) 
	
	#data proprocessing
    def preprocess(self, input_file):
        
        with open(input_file, 'r') as f:
            str = f.read()
        words = str.split()
        print "file read"
        self.dic_wrd = dict()
        self.data = list()
        count = [['UNK', -1]]
        frq_word = Counter(words)
        frq_word = frq_word.most_common(vocab_size - 1)
        count.extend(frq_word)

        for word, _ in count:
            self.dic_wrd[word] = len(self.dic_wrd)
        self.vocab_size = len(self.dic_wrd)
        print "counter words"
        for word in words:
            idx = 0
            if word in self.dic_wrd :
                idx = self.dic_wrd[word]
            self.data.append(idx)
        print "dict created"
        self.num_batches = int(len(self.data) / self.batch_size)
        #words
        self.reverse_dic_wrd = dict(zip(self.dic_wrd.values(), self.dic_wrd.keys()))
        
	#to get next batch
    def next_batch(self):
        
        if self.num_batches == 0:
            assert False, "number of batches is 0."
        batch_data = self.data[self.batch_ptr:self.batch_ptr+self.batch_size]
        self.batch_ptr = (self.batch_ptr+1)%self.num_batches
        xdata = list()
        ydata = list()
        for i in xrange(len(batch_data) - self.context_window_size):
            xdata.append(batch_data[i:i+self.context_window_size])
            ydata.append([batch_data[i+self.context_window_size]])
            #print batch_data[i:i+self.context_window_size]
            #print batch_data[i+self.context_window_size]
        #print  "batch number: ",self.batch_ptr
        return xdata,ydata




#calling for preprocessing
my_data = Corpusprocess(dir_corpus, sizeof_batch, window_size, file_name, vocab_size)
vocab_size = my_data.vocab_size

#setting up the graph and the computational unit for tensoflow
graph = tf.Graph()
with graph.as_default():
   #creating placeholder
    tf_in_data = tf.placeholder(tf.int32, [sizeof_batch-5, window_size])
    targets = tf.placeholder(tf.int64, [sizeof_batch-5, 1])

    #creating the variable scope for embeds
    with tf.variable_scope('nplmembed'):
        embeds = tf.Variable(tf.random_uniform([vocab_size, word_dim], -1.0, 1.0))
        embeds = tf.nn.l2_normalize(embeds, 1)

    with tf.variable_scope('nplmweight'):
    	#weight matrix variable for word embedding to hidden layer
    	#this variable is initialied with randow values
        weight_h = tf.Variable(tf.truncated_normal([window_size * word_dim + 1, hidden_l_sz],
                        stddev=1.0 / math.sqrt(hidden_l_sz)))
        #softmax variable
        softmax_w = tf.Variable(tf.truncated_normal([window_size * word_dim, vocab_size],
                        stddev=1.0 / math.sqrt(window_size * word_dim)))
        softmax_u = tf.Variable(tf.truncated_normal([hidden_l_sz + 1, vocab_size],
                        stddev=1.0 / math.sqrt(hidden_l_sz)))
	#lookup of nn  embedding and the with data input data 
    in_emb = tf.nn.embedding_lookup(embeds, tf_in_data)
    in_emb = tf.reshape(in_emb, [-1, window_size * word_dim])
    tf_pack = tf.pack([tf.shape(tf_in_data)[0], 1])
    tf_ones = tf.ones(tf_pack)
    in_emb_add = tf.concat(1, [in_emb, tf_ones])
	#tanh activation function
    inputs = tf.tanh(tf.matmul(in_emb_add, weight_h))
    inputs_add = tf.concat(1, [inputs, tf.ones(tf.pack([tf.shape(tf_in_data)[0], 1]))])
    #matmul for output
    opts = tf.matmul(inputs_add, softmax_u) + tf.matmul(in_emb, softmax_w)
    opts = tf.clip_by_value(opts, 0.0, grad_clip)
    #applying softmax
    opts = tf.nn.softmax(opts)
    onehot_tar = tf.one_hot(tf.squeeze(targets), vocab_size, 1.0, 0.0)
	#loss function and optimizer
    loss = -tf.reduce_mean(tf.reduce_sum(tf.log(opts) * onehot_tar , 1))
    optimizer = tf.train.AdagradOptimizer(alpha).minimize(loss)

    embeds_norm = tf.sqrt(tf.reduce_sum(tf.square(embeds), 1, keep_dims=True))
    norm_embed = embeds / embeds_norm

#running the graph in the session
with tf.Session(graph=graph) as sess:
	#initializing global variables
    tf.global_variables_initializer().run()
 	#running for only one epoch
    for b in range(my_data.num_batches):
        start = time.time()
        #get next batch
        x, y = my_data.next_batch()
        feed = {tf_in_data: x, targets: y}
        train_loss,  _ = sess.run([loss, optimizer], feed)
        end = time.time()
        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                b, my_data.num_batches,
                num_epochs-1, train_loss, end - start))
        np.save('wrd_embeds', norm_embed.eval())


