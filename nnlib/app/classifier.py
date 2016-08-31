#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.31
Instruction:

>> clf = Classifier()
>> clf.load_model(fname_model)
>> proba = clf.classify_proba(seqs_of_ids)
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle

import numpy as np
import theano
floatX = theano.config.floatX

import nnlib
from nnlib.common.dataset import BatchIterator

class Classifier:
	def __init__(self):
		self.input_adapt = nnlib.common.input_adapter.seqs2matrix_mask

	def classify_proba(self, seqs):
		DEFAULT_BATCH_SIZE = 32

		flag_single = not isinstance(seqs[0], list)

		if flag_single:
			seqs = [seqs, ]
			proba = self.model.predict(*self.input_adapt(seqs))
			return proba[0]
		else:
			iterator = BatchIterator(seqs)

			proba = []
			for seqs_batch in iterator.iterate(DEFAULT_BATCH_SIZE):
				proba.append(self.model.predict(*self.input_adapt(seqs_batch)))

			proba = np.concatenate(proba, axis = 0)
			return proba

	def load_model(self, fname_model):
		fname_config = '%s_config.pkl'%(fname_model)
		fname_param = '%s_param.npz'%(fname_model)

		model_config = cPickle.load(open(fname_config, 'r'))

		self.model = nnlib.model.get(model_config['model_name'])
		self.model.build(model_config)

		params = np.load(fname_param)
		self.model.update_tparams(params)


