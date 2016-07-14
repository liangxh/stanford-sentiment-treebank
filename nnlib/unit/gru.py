#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.09
Description: a factory for building the Gated Recurrent Unit
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

import common

def build(tparams, prefix, state_before, mask, dim, odim = None):
	if odim is None:
		odim = dim

	N = 3
	params = [
		('%s_W'%(prefix), np.concatenate([common.ortho_weight(dim, odim) for i in range(N)], axis=1)),
		('%s_U'%(prefix), np.concatenate([common.ortho_weight(odim, odim) for i in range(N)], axis=1)),
		('%s_b'%(prefix), np.zeros((N * odim,)).astype(floatX)),
		]

	for name, value in params:
		tparams[name] = theano.shared(value, name = name)

	nsteps = state_before.shape[0]

	if state_before.ndim == 3:
		n_samples = state_before.shape[1]
	else:
		n_samples = 1

	def _slice(_x, n, dim):
		if _x.ndim == 3:
			# for parallel computation
			return _x[:, :, n * dim:(n + 1) * dim]

		return _x[:, n * dim:(n + 1) * dim]

	def _step(x_, m_, h_):
		"""
		m_: mask
		x_: main input x
		h_: hidden output from the last loop
		"""

		preact = T.dot(h_, tparams['%s_U'%(prefix)])
		preact += x_

		i = T.nnet.sigmoid(_slice(preact, 0, odim))
		f = T.nnet.sigmoid(_slice(preact, 1, odim))
		g = T.tanh(_slice(preact, 2, odim))

		h = T.tanh(i * g + f * h_)
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


