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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def build_layer(state_before, flag, rate = 0.5, random_seed = None):
	trng = RandomStreams(default_seed) if random_seed is not None else RandomStreams()

	proj = T.switch(
			flag,
			(state_before
				* trng.binomial(
					state_before.shape,
					p = 1. - rate,
					dtype = state_before.dtype
					)
			),
			state_before * rate
		)

	return proj

