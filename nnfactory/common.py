#! /usr/bin/env
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.12
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

def ortho_weight(nrow, ncol = None, dtype = floatX):
	"""
	initialization of a matrix [nrow x ncol] with orthogonal weight
	"""
	dim = nrow if ncol is None else max(nrow, ncol)

	W = np.random.randn(dim, dim)
	u, s, v = np.linalg.svd(W)

	return u[:nrow, :ncol].astype(dtype)

def L2_norm(tparam):
	return (tparams['U'] ** 2).sum()


