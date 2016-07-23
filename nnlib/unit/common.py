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

import time
np.random.seed(int(time.time() * 1e6) % (1 << 32))

def rand_weight(shape, dtype = floatX):
	return 0.1 * np.random.randn(* shape).astype(dtype)

def ortho_weight(nrow, ncol = None, dtype = floatX):
	"""
	initialization of a matrix [nrow x ncol] with orthogonal weight
	"""
	dim = nrow if ncol is None else max(nrow, ncol)

	W = np.random.randn(dim, dim)
	u, s, v = np.linalg.svd(W)

	return u[:nrow, :ncol].astype(dtype)

def L2_norm(tparam):
	return (tparam ** 2).sum()


