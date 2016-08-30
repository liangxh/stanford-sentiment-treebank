#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.09
Description: a factory for building the Recurrent Neural Network layer
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

from nnlib.common import State, initializer

def build(tparams, prefix, state_before, mask, odim = None):
	dim = state_before.odim

	if odim is None:
		odim = dim

	params = [
		('%s_W'%(prefix), initializer.weight_orthogonal(dim, odim)),
		('%s_U'%(prefix), initializer.weight_orthogonal(odim, odim)),
		('%s_b'%(prefix), initializer.bias(odim)),
		]
	
	for name, value in params:
		tparams[name] = theano.shared(value, name = name)

	nsteps = state_before.var.shape[0]
	
	if state_before.var.ndim == 3:
		n_samples = state_before.var.shape[1]
	else:
		n_samples = 1

	def _step(x_, m_, h_):
		"""
		m_: mask
		x_: main input x
		h_: hidden output from the last loop
		"""

		#h = T.tanh(x_ + T.dot(h_, tparams['%s_U'%(prefix)]))
		h = T.nnet.sigmoid(x_ + T.dot(h_, tparams['%s_U'%(prefix)]))
		h = m_[:, None] * h + (1. - m_)[:, None] * h_ # cover h if m == 1

		return h

	var = T.dot(state_before.var, tparams['%s_W'%(prefix)]) + tparams['%s_b'%(prefix)]

	rval, updates = theano.scan(
				_step,
				sequences = [var, mask],
				outputs_info = [
					T.alloc(np.asarray(0., dtype = floatX), n_samples, odim),
					],
				name = '%s_layers'%(prefix),
				n_steps = nsteps
			)

	# hidden state output, memory states
	return State(rval, odim)

def postprocess_avg(state_before, mask):
	"""
	mean pooling
	
	proj: a matrix of size [n_step, n_samples, dim_proj]
	mask: a matrix of size [n_step, n_samples]
	"""

	var = (state_before.var * mask[:, :, None]).sum(axis=0)
	var = var / mask.sum(axis=0)[:, None]

	return State(var, state_before.odim)

def postprocess_last(state_before):
	"""
	keep only the last hidden state
	
	proj: a matrix of size [n_step, n_samples, dim_proj]
	"""

	return state_before.map(lambda k:k[-1])

