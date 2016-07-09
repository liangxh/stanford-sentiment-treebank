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
import theano.tensor as T
floatX = theano.config.floatX
#theano.config.exception_verbosity = 'high'
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Set the random number generators' seeds for consistency
SEED = 123 
np.random.seed(SEED)

###################### Shortcut ######################################
def np_floatX(data):
	return np.asarray(data, dtype = floatX)

###################### Zip and Unzip #################################
def zipp(params, tparams):
	'''
	set values for tparams using params
	'''
	for kk, vv in params.iteritems():
		tparams[kk].set_value(vv)

def unzip(zipped):
	'''
	get value from tparams to an OrderedDict for saving
	'''
	new_params = OrderedDict()
	for kk, vv in zipped.iteritems():
		new_params[kk] = vv.get_value()
	return new_params

##################### Tools ##################################################
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

def load_params(path, params):
	pp = np.load(path)
	for kk, vv in params.iteritems():
		if kk not in pp:
			raise Warning('%s is not in the archive' % kk)
		params[kk] = pp[kk]

	return params

def init_tparams(params):
	tparams = OrderedDict()
	for kk, pp in params.iteritems():
		tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams

import nnfactory

class Classifier:
	def __init__(self):
		pass

	########################## Building Model ###################################
	def init_params(self, options, Wemb = None):
		'''
		initizalize params for every layer
		'''
		params = OrderedDict()

		params['Wemb'] = Wemb

		def add_params(new_params):
			for k, v in new_params:
				params[k] = v
		
		add_params(nnfactory.gru.init_param('gru', options['dim_wemb'], options['dim_hidden']))
		add_params(nnfactory.rnn.init_param('rnn', options['dim_hidden'], options['dim_hidden']))
		add_params(nnfactory.softmax.init_param('softmax', options['dim_hidden'], options['ydim']))

		return params

	def build_model(self, tparams, options):

		# global flag for dropout
		flag_dropout = theano.shared(np_floatX(0.))

		# a matrix, whose shape is (n_timestep, n_samples) for theano.scan in training process
		x = T.matrix('x', dtype = 'int64')
		
		# a matrix, used to distinguish the valid elements in x
		mask = T.matrix('mask', dtype = floatX)

		# a vector of targets for $n_samples samples   
		y = T.vector('y', dtype = 'int64')

		n_timesteps = x.shape[0]
		n_samples = x.shape[1]

		# transfer x, the matrix of tids, into Wemb, the 'matrix' of embedding vectors 
		emb = tparams['Wemb'][x.flatten()].reshape(
				[n_timesteps, n_samples, options['dim_wemb']]
			)

		# the result of LSTM, a matrix of shape (n_timestep, n_samples, dim_hidden)
		proj = nnfactory.gru.build_layer(tparams, 'gru', emb, mask)
		proj = nnfactory.rnn.build_layer(tparams, 'rnn', proj, mask)

		proj = nnfactory.rnn.postprocess_last(proj)

		proj = nnfactory.dropout.build_layer(proj, flag_dropout)

		pred = nnfactory.softmax.build_layer(tparams, 'softmax', proj)

		# build functions for prediction
		f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
		f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

		# build function for cost calculation
		off = 1e-8
		if pred.dtype == 'float16':
			off = 1e-6

		cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

		# return flag_dropout, x, mask, y, f_pred_prob, f_pred, cost

		if options['decay_c'] > 0.:
			decay_c = theano.shared(np_floatX(options['decay_c']), name='decay_c')
			weight_decay = 0.
			weight_decay += (tparams['U'] ** 2).sum()
			weight_decay *= decay_c
			cost += weight_decay
	
		f_cost = theano.function([x, mask, y], cost, name = 'f_cost')
		
		grads = T.grad(cost, wrt = tparams.values())
		#f_grad = theano.function([x, mask, y], grads, name = 'f_grad')

		lr = T.scalar(name = 'lr')
		f_grad_shared, f_update = nnfactory.adadelta.build(lr, tparams, grads, [x, mask], y, cost)

		return f_pred, f_pred_prob, f_grad_shared, f_update, flag_dropout

	########################## Classification ###################################

	def load(self, fname_model):
		model_options = locals().copy()

		fname_config = '%s_config.pkl'%(fname_model)
		fname_param = '%s_param.npz'%(fname_model)

		train_params = cPickle.load(open(fname_config, 'r')) # why -1??
		model_options.update(train_params)

		params = self.init_params(model_options, None)
		load_params(fname_param, params)
		tparams = init_tparams(params)

		self.f_pred, self.f_pred_prob, f_grad_shared, f_update, flag_dropout = self.build_model(tparams, model_options)

	def predict_proba(self, seqs, batch_size = 64):

		def _predict(seqs):
			x, x_mask = self.prepare_x(seqs)
			proba = self.f_pred_prob(x, x_mask)
			
			return proba

		if not isinstance(seqs[0], list):
			seqs = [seqs, ]
			pred_probs = _predict(seqs)

			return pred_probs[0]

		elif batch_size is None:
			proba = _predict(seqs)
			return proba
		else:
			kf = get_minibatches_idx(len(seqs), batch_size)
			proba = []

			for _, idx in kf:
				proba.extend(_predict(np.asarray([seqs[i] for i in idx])))
			
			proba = np.asarray(proba)
			return proba

	def pred_error(self, data, iterator = None):
		data_x = data[0]
		data_y = np.array(data[1])
		valid_err = 0

		if iterator is None:
			iterator = get_minibatches_idx(len(data[0]), 32)

		for _, valid_index in iterator:
			x, mask = self.prepare_x([data_x[t] for t in valid_index])
			y = data_y[valid_index]

			preds = self.f_pred(x, mask)
			valid_err += (preds == y).sum()
			
		valid_err = 1. - np_floatX(valid_err) / len(data[0])

		return valid_err

	######################## Training ##########################################

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
	
	def train(self,
		dataset, Wemb, 
		
		# model params		
		reload_model = False,
		fname_model = None,
		
		# training params
		validFreq = 1000,
		saveFreq = 1000,
		patience = 10,
		max_epochs = 5000,
		decay_c = 0.,
		lrate = 0.0001,
		batch_size = 16,
		valid_batch_size = 64,
		noise_std = 0., 

		# debug params
		dispFreq = 100,
	):

		# preparing model
		fname_config = '%s_config.pkl'%(fname_model)
		fname_param = '%s_param.npz'%(fname_model)

		# building model		
		ydim = np.max(dataset[0][1]) + 1
		dim_wemb = Wemb.shape[1] # np.ndarray expected
		dim_hidden = dim_wemb

		# saving configuration of the model
		model_config = {
			'ydim':ydim,
			'dim_wemb':dim_wemb,
			'dim_hidden':dim_hidden,
			'fname_model':fname_model,
		}
		cPickle.dump(model_config, open(fname_config, 'wb'), -1) # why -1??

		# preparing options for building model 
		model_options = locals().copy()
		model_options['dim_wemb'] = dim_wemb
		model_options['dim_hidden'] = dim_hidden	
		params = self.init_params(model_options, Wemb)

		if reload_model:
			if os.path.exists(fname_param):
				load_params(fname_param, params)
			else:
				print >> sys.stderr, 'train: [warning] model %s not found'%(fname_param)
				return None
		elif Wemb is None:
			print >> sys.stderr, 'train: [warning] Wemb is missing for training LSTM'
			return None
		
		tparams = init_tparams(params)
		
		self.f_pred, self.f_pred_prob, f_grad_shared, f_update, flag_dropout = self.build_model(tparams, model_options)
		
		# preparing functions for training
		print >> sys.stderr, 'train: [info] preparing functions'

		# preparing
		train, valid, test = dataset

		kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
		kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
		
		history_errs = []
		best_p = None
		bad_count = 0

		uidx = 0       # number of update done
		estop = False  # early stop

		# training
		print >> sys.stderr, 'train: [info] start training...'

		start_time = time.time()

		try:
			for eidx in xrange(max_epochs):
				n_samples = 0
				
				kf = get_minibatches_idx(len(train[0]), batch_size, shuffle = True)
				
				for _, train_index in kf:
					uidx += 1
					flag_dropout.set_value(1.)

					x = [train[0][t] for t in train_index]
					y = [train[1][t] for t in train_index]

					x, mask = self.prepare_x(x)
					n_samples += x.shape[1]

					cost = f_grad_shared(x, mask, y)
					f_update(lrate)
					
					if np.isnan(cost) or np.isinf(cost):
						'''
						NaN of Inf encountered
						'''
						print >> sys.stderr, 'train: [warning] NaN detected'
						return 1., 1., 1.
					
					if np.mod(uidx, dispFreq) == 0:
						'''
						display progress at $dispFreq
						'''
						print >> sys.stderr, 'train: [info] Epoch %d Update %d Cost %f'%(eidx, uidx, cost)

					if np.mod(uidx, saveFreq) == 0:
						'''
						save new model to file at $saveFreq
						'''
						print >> sys.stderr, 'train: [info] Model update'
						
						if best_p is not None:
							params = best_p
						else:
							params = unzip(tparams)
					
						np.savez(fname_param, history_errs = history_errs, **params)

					if np.mod(uidx, validFreq) == 0:
						'''
						check prediction error at %validFreq
						'''
						flag_dropout.set_value(0.)
						
						print >> sys.stderr, 'train: [info] Validation ....'

						# not necessary	
						train_err = self.pred_error(train, kf)
						valid_err = self.pred_error(valid, kf_valid)
						test_err = self.pred_error(test, kf_test)
						
						history_errs.append([valid_err, test_err])
						if (uidx == 0 or valid_err <= np.array(history_errs)[:, 0].min()):
							best_p = unzip(tparams)
							bad_count = 0
						
						print >> sys.stderr, 'train: [info] precision: train %f valid %f test %f'%(
								1. - train_err, 1. - valid_err, 1. - test_err)
							
						if (len(history_errs) > patience and
							valid_err >= np.array(history_errs)[:-patience, 0].min()):
							bad_count += 1
							if bad_count > patience:
								print >> sys.stderr, 'train: [info] Early stop!'
								estop = True
								break

				print >> sys.stderr, 'train: [info] %d samples seen'%(n_samples)
				if estop:
					break
	
		except KeyboardInterrupt:
			print >> sys.stderr, 'train: [debug] training interrupted by user'

		end_time = time.time()

		if best_p is not None:
			zipp(best_p, tparams)
		else:
			best_p = unzip(tparams)

		flag_dropout.set_value(0.)
		
		kf_train = get_minibatches_idx(len(train[0]), batch_size)
		train_err = self.pred_error(train, kf_train)
		valid_err = self.pred_error(valid, kf_valid)
		test_err = self.pred_error(test, kf_test)
 
		print >> sys.stderr, 'train: [info] precision: train %f valid %f test %f'%(
				1. - train_err, 1. - valid_err, 1. - test_err)
		
		np.savez(
			fname_param,
			train_err = train_err,
			valid_err = valid_err,
			test_error = test_err,
			history_errs = history_errs, **best_p
			)

		print >> sys.stderr, 'train: [info] totally %d epoches in %.1f sec'%(eidx + 1, end_time - start_time)

		self.tparams = tparams

		return train_err, valid_err, test_err

import cPickle
import nltk
from wordembedder import WordEmbedder

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

def main():	
	optparser = OptionParser()

	optparser.add_option('-p', '--prefix', action='store', dest='prefix')
	optparser.add_option('-i', '--input', action='store', dest='key_input')
	optparser.add_option('-e', '--embed', action='store', dest='key_embed')
	optparser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)

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

	Wemb = wembedder.get_Wemb()

	print >> sys.stderr, 'main: [info] start training'
	clf = Classifier()

	res = clf.train(
			dataset = dataset,
			Wemb = Wemb,
			fname_model = fname_model,
			batch_size = opts.batch_size,
		)

	test_x, test_y = dataset[2]

	proba = clf.predict_proba(test_x)
	cPickle.dump((test_y, proba), open(fname_test, 'w'))	

	prec = precision_at_n(test_y, proba)
	cPickle.dump(prec, open(fname_prec, 'w'))

if __name__ == '__main__':
	main()
