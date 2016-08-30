#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.08
Description: a factory for building dropout layer
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams

import time
default_seed = int(time.time() * 1e6) % 2147462579

from nnlib.common import State

def build(state_before, flag, rate = 0.5):
	trng = MRG_RandomStreams(default_seed)

	var = T.switch(
			flag,
			(state_before.var
				* trng.binomial(
					state_before.var.shape,
					p = rate,
					dtype = state_before.var.dtype
					)
			),
			state_before.var * rate
		)

	return State(var, state_before.odim)

