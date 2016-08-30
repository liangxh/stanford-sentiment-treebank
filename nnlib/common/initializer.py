#! /usr/bin/env
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.31
'''

import math
import numpy as np
import theano
floatX = theano.config.floatX

import time
np.random.seed(int(time.time() * 1e6) % (1 << 32))

def weight(n_rol, n_col, dtype = floatX):
	return np.random.random([n_rol, n_col]).astype(dtype)

def weight_orthogonal(n_row, n_col, dtype = floatX):
	dim = max(n_row, n_col)

	W = np.random.randn(dim, dim)
	u, s, v = np.linalg.svd(W)

	return u[:n_row, :n_col].astype(dtype)

def weight_xavier(n_rol, n_col, dtype = floatX):
	span = math.sqrt(6./(n_rol + n_col))

	return np.random.uniform(-span, span, [n_rol, n_col])

def bias(dim, init = .1, dtype = floatX):
	if init == 0.:
		return np.zeros([dim, ], dtype = dtype)
	else:
		return np.ones([dim, ], dtype = dtype)

def bias_rand(dim, dtype = floatX):
	return np.random.random([dim, ]).astype(dtype)
