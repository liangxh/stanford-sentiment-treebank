#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.30
'''

import sys
import time
import cPickle
import numpy as np

import nnlib
from nnlib.common.dataset import DataPair, DataSet

class Trainer:
	def __init__(self):
		self.input_adapt = nnlib.common.input_adapter.seqs2matrix_mask

	def predict_error(self, data_iterator, batch_size = 32):
		err = 0

		for x, y in data_iterator.iterate(batch_size):
			proba = self.model.predict(self.input_adapt(x))
			err += (np.argmax(proba, axis = 1) == np.asarray(y)).sum()
			
		err = 1. - float(err) / data_iterator.num

		return err

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
		learning_rate = 0.0001,
		batch_size = 16,
		dim_hidden = None,

		valid_batch_size = 64,
		validFreq = 100,
		saveFreq = 100,
		patience = 10,
		max_epochs = 5000,

		# debug params
		dispFreq = 100,
	):
		if self.input_adapt is None:
			raise Warning('please set function for input adaptation before training (see common.input_adapter)')

		assert fname_model is not None

		# preparing model
		fname_config = '%s_config.pkl'%(fname_model)
		fname_param = '%s_param.npz'%(fname_model)

		if not resume:
			# building model
			ydim = dataset.ydim
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

		self.model = nnlib.model.get(model_config['model_name'])
		self.model.build(model_config)

		history_errs = []
		best_p = None
		best_p_updated = False
		bad_count = 0

		if not resume:
			if Wemb is None:
				print >> sys.stderr, 'train: [warning] Wemb is required for training'%(fname_param)
				return None
			else:
				self.model.load_value('Wemb', Wemb)
		else:
			if not os.path.exists(fname_param):
				print >> sys.stderr, 'train: [warning] model %s not found'%(fname_param)
				return None
			else:
				params = np.load(fname_param)
				self.model.update_tparams(params)
				best_p = self.model.get_params()
				
				if 'history_errs' in params:
					history_errs = params['history_errs'].tolist()

				print >> sys.stderr, 'train: [info] model loaded from %s'%(fname_param)				

		# training
		print >> sys.stderr, 'train: [info] start training...'
		start_time = time.time()
		uidx = 0
		estop = False

		try:
			for eidx in xrange(max_epochs):
				for x, y in dataset.train.iterate(batch_size, shuffle = True):
					uidx += 1

					cost = self.model.train(self.input_adapt(x), np.asarray(y), learning_rate)
					
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
						#flag_dropout.set_value(0.)
						
						print >> sys.stderr, 'train: [info] validation'

						#train_err = self.predict_error(train, kf)

						valid_err = self.predict_error(dataset.valid)
						test_err = self.predict_error(dataset.test)
						
						history_errs.append([valid_err, test_err])

						if (best_p is None) or (valid_err <= np.array(history_errs)[:, 0].min()):
							'''
							update best_p if params perform the best so far
							'''
							best_p = self.model.get_params()
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
			self.model.update_tparams(best_p)
		else:
			best_p = self.model.get_params()

		#flag_dropout.set_value(0.)
		
		train_err = self.predict_error(dataset.train)
		valid_err = self.predict_error(dataset.valid)
		test_err = self.predict_error(dataset.test)
 
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

