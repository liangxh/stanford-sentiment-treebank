#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.25
Description: a factory for building a fully connected layer (linear combination)
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

from nnlib.common import State, initializer

def build(tparams, prefix, states_before, odim, bias = False, bias_init = 0.):
	if not isinstance(states_before, list):
		states_before = [states_before, ]
	elif len(states_before) == 0:
		raise Warning("$states_before is an empty list")

	ndim = states_before[0].var.ndim

	value_concat = T.concatenate(map(lambda k:k.var, states_before), axis = ndim - 1)
	dim = np.sum(map(lambda k:k.odim, states_before))
	
	name = '%s_W'%(prefix)
	value_init = initializer.weight_xavier(dim, odim, floatX)
	tparams[name] = theano.shared(value_init, name = name)

	value = T.dot(value_concat, tparams[name])

	if bias:
		name = '%s_b'%(prefix)
		value_init = initializer.bias(odim, bias_init)
		tparams[name] = theano.shared(value_init, name = name)

		value += tparams[name]

	return State(value, odim)

