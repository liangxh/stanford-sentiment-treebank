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

import common

def build(tparams, prefix, state_before, mask, dim, odim = None):
	if odim is None:
		odim = dim

	params = [
		('%s_W'%(prefix), common.ortho_weight(dim, odim)),
		('%s_U'%(prefix), common.ortho_weight(odim, odim)),
		('%s_b'%(prefix), common.rand_weight((odim,), floatX)),
		]
	
	for name, value in params:
		tparams[name] = theano.shared(value, name = name)

	nsteps = state_before.shape[0]
	
	if state_before.ndim == 3:
		n_samples = state_before.shape[1]
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

	proj = T.dot(state_before, tparams['%s_W'%(prefix)]) + tparams['%s_b'%(prefix)]

	rval, updates = theano.scan(
				_step,
				sequences = [proj, mask],
				outputs_info = [
					T.alloc(np.asarray(0., dtype = floatX), n_samples, odim),
					],
				name = '%s_layers'%(prefix),
				n_steps = nsteps
			)

	# hidden state output, memory states
	return rval

def postprocess_avg(proj, mask):
	"""
	mean pooling
	
	proj: a matrix of size [n_step, n_samples, dim_proj]
	mask: a matrix of size [n_step, n_samples]
	"""

	proj = (proj * mask[:, :, None]).sum(axis=0)
	proj = proj / mask.sum(axis=0)[:, None]

	return proj

def postprocess_last(proj):
	"""
	keep only the last hidden state
	
	proj: a matrix of size [n_step, n_samples, dim_proj]
	"""

	return proj[-1]

