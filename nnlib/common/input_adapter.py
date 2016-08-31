#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.31
'''

import time
import theano
import numpy as np

floatX = theano.config.floatX

def seqs2matrix_mask(seqs):
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

