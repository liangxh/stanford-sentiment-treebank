#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.09
Description: a factory for building the Bidirectional Recurrent Neural Network layer
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

import rnn

def ortho_weight(nrow, ncol = None, dtype = floatX):
	"""
	initialization of a matrix [nrow x ncol] with orthogonal weight
	"""
	dim = nrow if ncol is None else max(nrow, ncol)

	W = np.random.randn(dim, dim)
	u, s, v = np.linalg.svd(W)

	return u[:nrow, :ncol].astype(dtype)


def init_param(rnn_module, prefix, dim, odim = None):
	if odim is None:
		odim = dim

	params = []
	params.extend(rnn_module.init_param('%s_fw'%(prefix), dim, odim))
	params.extend(rnn_module.init_param('%s_bw'%(prefix), dim, odim))

	params.append(('%s_Wf'%(prefix), ortho_weight(odim, odim)))
	params.append(('%s_Wb'%(prefix), ortho_weight(odim, odim)))
	params.append(('%s_b'%(prefix), np.zeros((odim,)).astype(floatX)))

	return params

def build_layer(
		rnn_module,

		# data access
		tparams,
		prefix,

		# params of the network
		state_below,
		mask,
	):

	proj_fw = rnn_module.build_layer(tparams, '%s_fw'%(prefix), state_below, mask)
	proj_bw = rnn_module.build_layer(tparams, '%s_bw'%(prefix), state_below[::-1], mask[::-1])[::-1]

	proj = T.dot(proj_fw, tparams['%s_Wf'%(prefix)]) + T.dot(proj_bw, tparams['%s_Wb'%(prefix)]) + tparams['%s_b'%(prefix)]

	return proj

