#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.09
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
import cPickle
from collections import OrderedDict
from optparse import OptionParser

import numpy as np
import theano
floatX = theano.config.floatX

import nltk
from wordembedder import WordEmbedder

import nnlib

def update_tparams(tparams, params):
	'''
	set values for tparams using params
	'''
	for name, value in params.iteritems():
		if name in tparams:
			tparams[name].set_value(value)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
	'''
	get batches of idx for range(1, n) and shuffle if needed
	'''
	idx_list = np.arange(n, dtype="int32")

	if shuffle:
		np.random.shuffle(idx_list)

	minibatches = []
	minibatch_start = 0
	for i in range(n // minibatch_size):
		minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
		minibatch_start += minibatch_size

	if (minibatch_start != n):
		# Make a minibatch out of what is left
		minibatches.append(idx_list[minibatch_start:])

	return zip(range(len(minibatches)), minibatches)

class Classifier:
	def __init__(self):
		pass

	def predict_error(self, data, iterator = None):
		data_x = data[0]
		data_y = np.array(data[1])
		err = 0

		if iterator is None:
			iterator = get_minibatches_idx(len(data[0]), 32)

		for _, valid_index in iterator:
			x, mask = self.prepare_x([data_x[t] for t in valid_index])
			y = data_y[valid_index]

			preds = self.f_pred(x, mask)
			err += (preds == y).sum()
			
		err = 1. - float(err) / len(data[0])

		return err

	def prepare_x(self, seqs):
		'''
		create two 2D-Arrays (seqs and mask)
		'''
		lengths = [len(s) for s in seqs]

		n_samples = len(seqs)
		maxlen = np.max(lengths)

		x = np.zeros((maxlen, n_samples)).astype('int64')
		x_mask = np.zeros((maxlen, n_samples)).astype(floatX)

		for idx, s in enumerate(seqs):
			x[:lengths[idx], idx] = s
			x_mask[:lengths[idx], idx] = 1.
		
		return x, x_mask

	def load_model(self, fname_model):
		fname_config = '%s_config.pkl'%(fname_model)
		fname_param = '%s_param.npz'%(fname_model)

		model_config = cPickle.load(open(fname_config, 'r'))

		model = nnlib.model.get(model_config['model_name'])

		tparams, self.f_pred, self.f_pred_prob, \
			f_grad_shared, f_update, flag_dropout = model.build(model_config)

		params = np.load(fname_param)
		update_tparams(tparams, params)

		return params

def main():
	optparser = OptionParser()

	optparser.add_option('-p', '--prefix', action='store', dest='prefix')
	optparser.add_option('-i', '--input', action='store', dest='key_input')
	optparser.add_option('-e', '--embed', action='store', dest='key_embed')
	optparser.add_option('-m', '--model', action='store', dest='model_name')
	
	opts, args = optparser.parse_args()

	prefix = opts.prefix
	fname_input = 'data/dataset/' + '%s.pkl'%(opts.key_input)
	fname_embed = 'data/wemb/' + '%s.txt'%(opts.key_embed)

	fname_model = 'data/model/' + '%s'%(prefix)
	fname_test = 'data/test/' + '%s_test.pkl'%(prefix)
	fname_prec = 'data/test/' + '%s_prec.pkl'%(prefix)

	dataset = cPickle.load(open(fname_input, 'r'))
	wembedder = WordEmbedder.load(fname_embed)

	def preprocess_text(wembedder, xy):
		texts, y = xy
		seqs = [nltk.word_tokenize(t.lower()) for t in texts]
		idxs = [wembedder.seq2idx(seq) for seq in seqs]

		return (idxs, y)
	
	test = preprocess_text(wembedder, dataset[2])

	clf = Classifier()
	clf.load_model(fname_model)

	print 1. - clf.predict_error(test)

if __name__ == '__main__':
	main()
