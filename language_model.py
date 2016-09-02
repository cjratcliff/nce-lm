from __future__ import division

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L

import pickle
import random
import string
from collections import Counter

# Based on https://github.com/ebenolson/pydata2015/blob/master/4%20-%20Recurrent%20Networks/RNN%20Character%20Model%20-%202%20Layer.ipynb

# Options
seq_length = 50
batch_size = 32
max_iterations = 20000
rnn_size = 512
dropout_prob = 0.1
# Defines two layers
primer_length = 10 # Words
vocab_size = 10000 # 9999 and an unknown word token

K = 10
Z = pow(np.e,9)

# For stability during training, gradients are clipped and a total gradient norm constraint is also used
clip_gradients = True
max_grad_norm = 15

summary_freq = 10
val_freq = 50
load_model = False
save_model = True # Save the parameters and vocabulary
model_path = 'models/rnn.pkl'


class NCELayer(L.DenseLayer):
    def __init__(self, incoming, num_units, Z, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), **kwargs):
        super(L.DenseLayer, self).__init__(incoming, **kwargs)

        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape): ### already inherited?
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input,self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return T.exp(activation)/Z
		

class NoiseDist:
	# Unigram priors
	def __init__(self, frequencies, vocabulary):
		self.dist = {}
		
		total = np.sum([i[1] for i in frequencies])
		for i in range(len(frequencies)):
			self.dist[vocabulary[frequencies[i][0]]] = frequencies[i][1]/total
			
		self.dist_np = np.array(self.dist.values())
		self.dist_np = T.reshape(self.dist_np,(vocab_size,))

	# Returns [(w1,p(w1)),(w2,p(w2)),...,(wk,p(wk))]
	def sample(self, k):
		words = np.random.choice(self.dist.keys(),k,p=self.dist.values())
		return ([self.dist[word] for word in words],words)		
		

# Yields sequential portions of the corpus of size seq_length,
# starting from random locations and wrapping around the end of the data.
def data_batch_generator(corpus, size=batch_size):
	startidx = np.random.randint(0, len(corpus) - seq_length - 1, size=size)
	while True:
		items = np.array([corpus[start:start + seq_length + 1] for start in startidx])
		startidx = (startidx + seq_length) % (len(corpus) - seq_length - 1)
		yield items
		
# After sampling a data batch, we transform it into a one hot feature representation
# Create a target sequence by shifting by one character
def prep_batch_for_network(batch, vocabulary, vocab_size, seq_length):
	x_seq = np.zeros((len(batch), seq_length), dtype='int32')
	y_seq = np.zeros((len(batch), seq_length), dtype='int32')

	for i, item in enumerate(batch):
		for j in range(seq_length):
			if item[j] in vocabulary.keys():
				x_seq[i,j] = vocabulary[item[j]]
			else:
				x_seq[i,j] = vocabulary['UNK']
				
			if item[j+1] in vocabulary.keys():
				y_seq[i,j] = vocabulary[item[j+1]]
			else:
				y_seq[i,j] = vocabulary['UNK']
				
	return x_seq, y_seq
	
# We flatten the sequence into the batch dimension before calculating the loss
def calc_cross_ent(net_output, targets, vocab_size):
	preds = T.reshape(net_output, (-1, vocab_size))
	targets = T.flatten(targets)
	cost = T.nnet.categorical_crossentropy(preds, targets)
	return cost
	

def build_rnn(x_sym, hid_init_sym, hid2_init_sym, seq_length, vocab_size, rnn_size):

	l_input = L.InputLayer(input_var=x_sym, shape=(None, seq_length))
	l_input_hid = L.InputLayer(input_var=hid_init_sym, shape=(None, rnn_size))
	l_input_hid2 = L.InputLayer(input_var=hid2_init_sym, shape=(None, rnn_size))
	
	l_input = L.EmbeddingLayer(l_input, input_size=vocab_size, output_size=rnn_size)

	l_rnn = L.LSTMLayer(l_input, num_units=rnn_size, hid_init=l_input_hid)#, cell_init=l_init_cell)
	h = L.DropoutLayer(l_rnn,p=dropout_prob)
	l_rnn2 = L.LSTMLayer(h, num_units=rnn_size, hid_init=l_input_hid2)#, cell_init=l_init_cell2)
	h = L.DropoutLayer(l_rnn2,p=dropout_prob)

	# Before the decoder layer, we need to reshape the sequence into the batch dimension,
	# so that timesteps are decoded independently.
	l_shp = L.ReshapeLayer(h, (-1, rnn_size))
	
	pred = NCELayer(l_shp, num_units=vocab_size, Z=Z)
	pred = L.ReshapeLayer(pred, (-1, seq_length, vocab_size))
	return l_rnn, l_rnn2, pred
	
	
def preprocess_string(s):
	# s is a corpus of text as a single string
	# Split into a list of words
	X = s.split(' ') 
	# Remove any non alphabetic characters
	X = [''.join(e for e in i if e.isalpha() or e in string.whitespace) for i in X]
	# Remove newlines and tabs
	X = [i.rstrip() for i in X]
	# Convert to lowercase
	X = [i.lower() for i in X]
	X = [i.replace('\n','') for i in X]
	# Remove zero-length strings
	X = [i for i in X if i != '']
	return X


def create_indices(n,m):
	indices = np.arange(n,dtype='int32')
	indices = np.stack([indices for i in range(m)],axis=1)
	indices = np.ndarray.flatten(indices)
	return indices	
	
	
def main():
	# Load the corpus - 1.69m words
	print "Loading corpus..."
	corpus = open('tiny-shakespeare.txt','r').read()
	print "Processing corpus..."
	corpus = preprocess_string(corpus)
	
	if load_model:
		print "Loading vocabulary..."
		d = pickle.load(open(model_path, 'r'))
		vocabulary = d['vocabulary']
	else:
		print "Creating vocabulary..."
		# Replace each word with a number if it is in the vocab_size most frequent words
		frequencies = Counter(corpus)
		print "Total words: ", len(frequencies)
		frequencies = frequencies.most_common(vocab_size-1) # Subtract 1 to make room for UNK
	
		vocabulary = {}
		index = 0
		for word,_ in frequencies:
			vocabulary[word] = index
			index += 1
		
		vocabulary['UNK'] = vocab_size - 1
		frequencies.append(('UNK',20)) ###
		
	noise_dist = NoiseDist(frequencies, vocabulary)
		
	inv_vocabulary = {v:k for k,v in vocabulary.items()}

	# Reserve 10% of the data for validation
	train_corpus = corpus[:(len(corpus) * 9 // 10)]
	val_corpus = corpus[(len(corpus) * 9 // 10):]
			
	# Symbolic variables for input. In addition to the usual features and target,
	# we need initial values for the RNN layer's hidden states
	x_sym = T.imatrix()
	y_sym = T.imatrix()
	hid_init_sym = T.matrix()
	hid2_init_sym = T.matrix()
	p_rnn = T.scalar()
	noise_word_indices = T.ivector()
	batch_indices = T.ivector()
	
	print "Building model..."
	l_rnn, l_rnn2, l_out = build_rnn(x_sym, hid_init_sym, hid2_init_sym, seq_length, vocab_size, rnn_size)
	
	# We extract the hidden state of each RNN layer as well as the output of the decoder.
	# Only the hidden state at the last timestep is needed
	hid_out, hid2_out, prob_out = L.get_output([l_rnn, l_rnn2, l_out])

	hid_out = hid_out[:,-1]
	hid2_out = hid2_out[:,-1]

	batch_indices = create_indices(batch_size,seq_length)
	seq_indices = create_indices(seq_length,batch_size)	

	p_rnn = prob_out[batch_indices,seq_indices,T.flatten(y_sym)] # (batch_size)
	p_rnn = T.reshape(p_rnn,(batch_size,seq_length))

	pn = noise_dist.dist_np[T.flatten(y_sym)]
	pn = T.reshape(pn,(batch_size,seq_length))
	pcrnn = p_rnn/(p_rnn + K*pn) # (batch_size)
	
	batch_indices = create_indices(batch_size, seq_length*K)
	seq_indices = create_indices(seq_length, batch_size*K)
	
	noise_sample = noise_dist.sample(batch_size*seq_length*K)
	p_n_wij = np.array(noise_sample[0],dtype='float32') # (batch_size*K)
	p_n_wij = T.reshape(p_n_wij, (batch_size,seq_length,K))
	noise_word_indices = np.array(noise_sample[1],dtype='int32') # (batch_size*K)
	p_n_wij *= K
	
	p_nce_wij = prob_out[batch_indices,seq_indices,noise_word_indices]
	p_nce_wij = T.reshape(p_nce_wij,(batch_size,seq_length,K))

	pcn_list = p_n_wij/(p_nce_wij + p_n_wij) # (batch_size,K)
	
	loss = -(T.log(pcrnn) + T.sum(T.log(pcn_list),axis=(2)))
	loss = T.mean(loss)

	all_params = L.get_all_params(l_out, trainable=True)
	all_grads = T.grad(loss, all_params)
	
	if clip_gradients:
		all_grads = [T.clip(g,-5,5) for g in all_grads]
		all_grads, norm = lasagne.updates.total_norm_constraint(all_grads, max_grad_norm, return_norm=True)

	updates = lasagne.updates.adam(all_grads, all_params)

	train_fn = theano.function([x_sym, y_sym, hid_init_sym, hid2_init_sym],
							  [loss, hid_out, hid2_out], updates=updates, on_unused_input='warn')

	val_fn = theano.function([x_sym, y_sym, hid_init_sym, hid2_init_sym], [loss, hid_out, hid2_out], on_unused_input='warn')

	hid = np.zeros((batch_size, rnn_size), dtype='float32')
	hid2 = np.zeros((batch_size, rnn_size), dtype='float32')

	# Each iteration is a random sub-sequence of seq_length words
	train_batch_gen = data_batch_generator(train_corpus)
	val_batch_gen = data_batch_generator(val_corpus)
	
	# Load pre-trained weights into network
	if load_model:
		print "Loading model..."
		d = pickle.load(open(model_path, 'r'))
		L.set_all_param_values(l_out, d['param values'])

	train_losses = []
	for iteration in range(max_iterations):
		x, y = prep_batch_for_network(next(train_batch_gen), vocabulary, vocab_size, seq_length)
		loss_train,_,_ = train_fn(x, y, hid, hid2) ### Update the hidden states
		train_losses.append(loss_train)

		if iteration % summary_freq == 0:
			print 'Iteration {}\tTraining loss: {}'.format(iteration, np.mean(train_losses))
			train_losses = []
			
		if iteration % val_freq == 0 and iteration > 0:
			x, y = prep_batch_for_network(next(val_batch_gen), vocabulary, vocab_size, seq_length)
			loss_val,_,_ = val_fn(x, y, hid, hid2)
			print '\t\tValidation loss: {}'.format(loss_val)

			param_values = L.get_all_param_values(l_out)
			d = {'param values': param_values,
				 'vocabulary': vocabulary, 
				}
			
			if save_model:
				path = "models/rnn_trained.pkl"
				pickle.dump(d, open(path,'w'), protocol=pickle.HIGHEST_PROTOCOL)

	predict_fn = theano.function([x_sym, hid_init_sym, hid2_init_sym], [prob_out, hid_out, hid2_out])

	# Calculate validation loss
	hid = np.zeros((batch_size, rnn_size), dtype='float32')
	hid2 = np.zeros((batch_size, rnn_size), dtype='float32')

	# For faster sampling, we rebuild the network with a sequence length of 1
	l_rnn, l_rnn2, l_out = build_rnn(x_sym, hid_init_sym, hid2_init_sym, 1, vocab_size, rnn_size)
												
	hid_out, hid2_out, prob_out = L.get_output([l_rnn, l_rnn2, l_out])
	
	hid_out = hid_out[:,-1]
	hid2_out = hid2_out[:,-1]
	prob_out = prob_out[0,-1]

	L.set_all_param_values(l_out, d['param values'])

	predict_fn = theano.function([x_sym, hid_init_sym, hid2_init_sym], [prob_out, hid_out, hid2_out])

	# We feed character one at a time from the priming sequence into the network.
	# To obtain a sample string, at each timestep we sample from the output probability distribution,
	# and feed the chosen character back into the network. We terminate after the first linebreak.
	hid = np.zeros((1, rnn_size), dtype='float32')
	hid2 = np.zeros((1, rnn_size), dtype='float32')
	x = np.zeros((1, 1), dtype='int32')

	# We will use random sentences from the validation corpus to 'prime' the network
	start = random.randint(0, len(val_corpus))
	primer = val_corpus[start: min(len(val_corpus)-1, start+primer_length)]

	# Feed the primer into the network
	for word in primer:
		p, hid, hid2 = predict_fn(x, hid, hid2)
		if word in vocabulary.keys():
			# Input to feed into the RNN
			x[0,0] = vocabulary[word]
		
	# Generate the new string (fixed length)
	str = ''
	for _ in range(50):
		p, hid, hid2 = predict_fn(x, hid, hid2)
		p = p/(1 + 1e-6)
		
		# Normalize probabilities
		p /= np.sum(p)
		
		# Draw a sample from the multinomial distribution
		s = np.random.multinomial(1,p)
		str += inv_vocabulary[s.argmax(-1)] + ' '
		x[0,0] = s.argmax(-1)
			
	print 'Primer: ' + ' '.join(primer)
	print 'Generated: ' + str

if __name__ == "__main__":
	main()
