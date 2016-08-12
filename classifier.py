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


def update_tparams(tparams, params):
	'''
	set values for tparams using params
	'''
	for name, value in tparams.iteritems():
		if not name in params:
			raise Warning('param %s not found'%(name))

		tparams[name].set_value(params[name])

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

	def predict_proba(self, seq):
		def _predict(seqs):
			x, x_mask = self.prepare_x(seqs)

			return self.f_pred_prob(x, x_mask)

		seqs = [seq, ]
		proba = _predict(seqs)

		return proba[0]


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


import nltk
from wordindexer import WordIndexer

import nnlib

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

	windexer = WordIndexer.load(fname_embed)
	
	clf = Classifier()
	clf.load_model(fname_model)

	s = 'hello, how are you?'
	seq = nltk.word_tokenize(s.lower())
	idx = windexer.seq2idx(seq)
	
	res = clf.predict_proba(idx)
	print res
	

if __name__ == '__main__':
	main()
