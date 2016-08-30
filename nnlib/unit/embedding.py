#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.09
Description: a factory for building embedding
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

from nnlib.common import State

def build(tparams, prefix, ids, vocab_size, dim):
	name = prefix
	value_init = np.zeros((vocab_size, dim), dtype = floatX)
	tparams[name] = theano.shared(value_init, name = name)

	new_shape = list(ids.shape) + [dim, ]
	
	var = tparams[name][ids.flatten()].reshape(new_shape)

	return State(var, dim)

