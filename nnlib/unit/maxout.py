#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.26
Description: a factory for build a maxout layer
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

import linear_combo
from nnlib.common import State

def build(tparams, prefix, states_before, odim, N = None, bias = False, bias_init = 0.):
	if N is None or not isinstance(N, int):
		raise Warning('$N of type int is required for maxout layer')

	if not isinstance(states_before, list):
		states_before = [states_before, ]
	elif len(states_before) == 0:
		raise Warning("$states_before is an empty list")

	batch_size = states_before[0].var.shape[0]

	var = linear_combo.build(tparams, prefix, states_before, odim *  N, bias, bias_init).var
	var = var.reshape([batch_size, odim, N])
	var = T.max(var, axis = var.ndim - 1)

	return State(var, odim)
