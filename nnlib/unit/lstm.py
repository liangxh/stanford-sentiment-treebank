#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.08
Description: a factory for building the Long Short Term Memory (LSTM) layer
'''

from nnlib.common.state import State

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

from nnlib.common import State, initializer

def build(tparams, prefix, state_before, mask, odim = None):
	dim = state_before.odim

	if odim is None:
		odim = dim

	N = 4
	params = [
		('%s_W'%(prefix), np.concatenate([initializer.weight_orthogonal(dim, odim) for i in range(N)], axis=1)),
		('%s_U'%(prefix), np.concatenate([initializer.weight_orthogonal(odim, odim) for i in range(N)], axis=1)),
		('%s_b'%(prefix), initializer.bias(N * odim)),
		]

	for name, value in params:
		tparams[name] = theano.shared(value, name = name)

	nsteps = state_before.var.shape[0]
	
	if state_before.var.ndim == 3:
		n_samples = state_before.var.shape[1]
	else:
		n_samples = 1

	def _slice(_x, n, dim):
		if _x.ndim == 3:
			return _x[:, :, n * dim:(n + 1) * dim]

		return _x[:, n * dim:(n + 1) * dim]

	def _step(x_, m_, h_, c_):
		"""
		m_: mask
		x_: main input x
		h_: hidden output from the last loop
		c_: memory in the cell from the last loop
		"""

		preact = T.dot(h_, tparams['%s_U'%(prefix)])
		preact += x_

		i = T.nnet.sigmoid(_slice(preact, 0, odim))
		f = T.nnet.sigmoid(_slice(preact, 1, odim))
		o = T.nnet.sigmoid(_slice(preact, 2, odim))
		c = T.tanh(_slice(preact, 3, odim))

		c = f * c_ + i * c
		c = m_[:, None] * c + (1. - m_)[:, None] * c_ # cover c if m == 1

		h = o * T.tanh(c)
		h = m_[:, None] * h + (1. - m_)[:, None] * h_ # cover h if m == 1

		return h, c

	var = (T.dot(state_before.var, tparams['%s_W'%(prefix)]) + tparams['%s_b'%(prefix)])

	rval, updates = theano.scan(
				_step,
				sequences = [var, mask],
				outputs_info = [
					# hidden state
					T.alloc(np.asarray(0., dtype = floatX), n_samples, odim),
					# memory cell
					T.alloc(np.asarray(0., dtype = floatX), n_samples, odim)
					],
				name = '%s_layers'%(prefix),
				n_steps = nsteps
			)

	return State(rval[0], odim)

