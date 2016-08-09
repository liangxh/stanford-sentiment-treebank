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
np.random.seed(int(time.time() * 1e6) % (1 << 32))

import theano
floatX = theano.config.floatX

def get_params(tparams):
	'''
	get value from tparams to an OrderedDict for saving
	'''
	params = OrderedDict()
	for name, value in tparams.iteritems():
		params[name] = value.get_value()

	return params

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

import nnlib

class Classifier:
	def __init__(self):
		pass

	def predict_proba(self, seqs, batch_size = 64):

		def _predict(seqs):
			x, x_mask = self.prepare_x(seqs)

			return self.f_pred_prob(x, x_mask)

		if not isinstance(seqs[0], list):
			seqs = [seqs, ]
			proba = _predict(seqs)

			return proba[0]
		else:
			kf = get_minibatches_idx(len(seqs), batch_size)
			proba = []

			for _, idx in kf:
				proba.extend(_predict(np.asarray([seqs[i] for i in idx])))
			
			proba = np.asarray(proba)
			return proba

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
	
	def train(
		self,
		dataset,
		Wemb = None, 

		# model params		
		fname_model = None,
		resume = False,
		model_name = None,
		
		# training params
		decay_c = 0.,
		lrate = 0.0001,
		batch_size = 16,
		dim_hidden = None,

		valid_batch_size = 64,
		validFreq = 1000,
		saveFreq = 1000,
		patience = 10,
		max_epochs = 5000,

		# debug params
		dispFreq = 100,
	):
		assert fname_model is not None

		# preparing model
		fname_config = '%s_config.pkl'%(fname_model)
		fname_param = '%s_param.npz'%(fname_model)

		if not resume:
			# building model		
			ydim = np.max(dataset[0][1]) + 1
			vocab_size = Wemb.shape[0]
			dim_wemb = Wemb.shape[1] # np.ndarray expected

			if dim_hidden is None:
				dim_hidden = dim_wemb
		
			# saving configuration of the model
			model_config = {
				'model_name': model_name,
				'ydim':ydim,
				'vocab_size':vocab_size,
				'dim_wemb':dim_wemb,
				'dim_hidden':dim_hidden,
				'decay_c':decay_c,
			}
			cPickle.dump(model_config, open(fname_config, 'wb'), -1) # why -1??
			print >> sys.stderr, 'train: [info] model configuration saved in %s'%(fname_config)
		else:
			model_config = cPickle.load(open(fname_config, 'r'))

		# preparing functions for training
		print >> sys.stderr, 'train: [info] building model...'

		model = nnlib.model.get(model_config['model_name'])

		tparams, self.f_pred, self.f_pred_prob, \
			f_grad_shared, f_update, flag_dropout = model.build(model_config)


		history_errs = []
		best_p = None
		best_p_updated = False
		bad_count = 0

		if not resume:
			if Wemb is None:
				print >> sys.stderr, 'train: [warning] Wemb is required for training'%(fname_param)
				return None
			else:
				tparams['Wemb'].set_value(Wemb)
		else:
			if not os.path.exists(fname_param):
				print >> sys.stderr, 'train: [warning] model %s not found'%(fname_param)
				return None
			else:
				params = np.load(fname_param)
				update_tparams(tparams, params)
				best_p = get_params(tparams)
				
				if 'history_errs' in params:
					history_errs = params['history_errs'].tolist()

				print >> sys.stderr, 'train: [info] model loaded from %s'%(fname_param)				

		# preparing
		train, valid, test = dataset

		kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
		kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
		

		# training
		print >> sys.stderr, 'train: [info] start training...'
		start_time = time.time()
		uidx = 0
		estop = False

		try:
			for eidx in xrange(max_epochs):
				kf = get_minibatches_idx(len(train[0]), batch_size, shuffle = True)
				
				for _, train_index in kf:
					uidx += 1
					flag_dropout.set_value(1.)

					x = [train[0][t] for t in train_index]
					y = [train[1][t] for t in train_index]

					x, mask = self.prepare_x(x)

					cost = f_grad_shared(x, mask, y)
					f_update(lrate)
					
					if np.isnan(cost) or np.isinf(cost):
						'''
						NaN of Inf encountered
						'''
						print >> sys.stderr, 'train: [warning] cost of NaN detected'
						return 1., 1., 1.
					
					if np.mod(uidx, dispFreq) == 0:
						'''
						display progress at $dispFreq
						'''
						print >> sys.stderr, 'train: [info] epoch %d update %d cost %f'%(eidx, uidx, cost)

					if np.mod(uidx, validFreq) == 0:
						'''
						check prediction error at %validFreq
						'''
						flag_dropout.set_value(0.)
						
						print >> sys.stderr, 'train: [info] validation'

						#train_err = self.predict_error(train, kf)

						valid_err = self.predict_error(valid, kf_valid)
						test_err = self.predict_error(test, kf_test)
						
						history_errs.append([valid_err, test_err])

						if (best_p is None) or (valid_err <= np.array(history_errs)[:, 0].min()):
							'''
							update best_p if params perform the best so far
							'''
							best_p = get_params(tparams)
							best_p_updated = True
							bad_count = 0
						
						print >> sys.stderr, 'train: [info] precision valid %f test %f'%(
									1. - valid_err, 1. - test_err)
							
						if (len(history_errs) > patience and
							valid_err >= np.array(history_errs)[:-patience, 0].min()):
							'''
							increase bad-count if the params is the best within the "patience window",
							'''
							bad_count += 1

							if bad_count > patience:
								print >> sys.stderr, 'train: [info] early stop!'
								estop = True
								break

					if np.mod(uidx, saveFreq) == 0 and best_p is not None and best_p_updated:
						'''
						save new model to file at $saveFreq
						'''
						np.savez(fname_param, history_errs = history_errs, **best_p)
						best_p_updated = False

						print >> sys.stderr, 'train: [info] model updated'

				if estop:
					break
	
		except KeyboardInterrupt:
			print >> sys.stderr, 'train: [debug] training interrupted by user'

		end_time = time.time()
		print >> sys.stderr, 'train: [info] totally %d epoches in %.1f sec'%(eidx + 1, end_time - start_time)

		# evaluate the model over the whole dataset
		if best_p is not None:
			update_tparams(tparams, best_p)
		else:
			best_p = get_params(tparams)

		flag_dropout.set_value(0.)
		
		kf_train = get_minibatches_idx(len(train[0]), batch_size)
		train_err = self.predict_error(train, kf_train)
		valid_err = self.predict_error(valid, kf_valid)
		test_err = self.predict_error(test, kf_test)
 
		print >> sys.stderr, 'train: [info] precision: train %f valid %f test %f'%(
				1. - train_err, 1. - valid_err, 1. - test_err)

		print >> sys.stderr, 'train: [info] model params saved to %s'%(fname_param)

		np.savez(
			fname_param,
			train_err = train_err,
			valid_err = valid_err,
			test_error = test_err,
			history_errs = history_errs,
			**best_p
			)

		self.tparams = tparams

		return train_err, valid_err, test_err

def precision_at_n(ys, pred_probs):
	n_test = len(ys)
	y_dim = len(pred_probs[0])
	hit = [0 for i in range(y_dim)]

	for y, probs in zip(ys, pred_probs):
		eid_prob = sorted(enumerate(probs), key = lambda k:-k[1])

		for i, item in enumerate(eid_prob):
			eid, progs = item
			if y == eid:
				hit[i] += 1

	for i in range(1, y_dim):
		hit[i] += hit[i - 1]

	prec = [float(hi) / n_test for hi in hit]
	return prec

import nltk
from wordembedder import WordEmbedder

def main():
	optparser = OptionParser()

	optparser.add_option('-p', '--prefix', action='store', dest='prefix')
	optparser.add_option('-i', '--input', action='store', dest='key_input')
	optparser.add_option('-e', '--embed', action='store', dest='key_embed')
	optparser.add_option('-m', '--model', action='store', dest='model_name')
	optparser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)

	optparser.add_option('-d', '--dim_hidden', action='store', dest='dim_hidden', type='int', default = None)
	optparser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)
	optparser.add_option('-l', '--learning_rate', action='store', type='float', dest='lrate', default = 0.05)
	optparser.add_option('-c', '--decay_c', action='store', type='float', dest='decay_c', default = 1e-4)
	
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
	
	dataset = [preprocess_text(wembedder, subset) for subset in dataset]
	dataset = tuple(dataset)

	if not opts.resume:
		Wemb = wembedder.get_Wemb()
	else:
		Wemb = None

	print >> sys.stderr, 'main: [info] start training'
	clf = Classifier()

	res = clf.train(
			dataset = dataset,
			Wemb = Wemb,

			fname_model = fname_model,
			resume = opts.resume,

			model_name = opts.model_name,
			dim_hidden = opts.dim_hidden,
			batch_size = opts.batch_size,
			decay_c = opts.decay_c,
			lrate = opts.lrate,
		)

	test_x, test_y = dataset[2]

	proba = clf.predict_proba(test_x)
	cPickle.dump((test_y, proba), open(fname_test, 'w'))	

	prec = precision_at_n(test_y, proba)
	cPickle.dump(prec, open(fname_prec, 'w'))

if __name__ == '__main__':
	main()
