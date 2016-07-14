
#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.08
Description: a factory for build Softmax layer
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

import common

def build_layer(tparams, prefix, state_before, dim, odim = None):
	if odim is None:
		odim = dim

	params = [
		('%s_U'%(prefix), common.ortho_weight(dim, odim, floatX)),
		('%s_b'%(prefix), common.rand_weight((odim, ), floatX)),
		]

	for name, value in params:
		tparams[name] = theano.shared(value, name = name)

	return T.nnet.softmax(T.dot(state_before, tparams['%s_U'%(prefix)]) + tparams['%s_b'%(prefix)])

	
