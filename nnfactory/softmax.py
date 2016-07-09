#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.08
Description: a factory for build Softmax layer
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

def init_param(prefix, dim, odim = None):
	if odim is None:
		odim = dim

	params = []
	params.append((
			'%s_U'%(prefix),
			0.01 * np.random.randn(dim, odim).astype(floatX)
		))
	params.append((
			'%s_b'%(prefix),
			np.zeros((odim,)).astype(floatX)
		))

	return params

def build_layer(
		# data access
		tparams,
		prefix,

		# input of the layer
		state_below,
	):

	return T.nnet.softmax(T.dot(state_below, tparams['%s_U'%(prefix)]) + tparams['%s_b'%(prefix)])

	
