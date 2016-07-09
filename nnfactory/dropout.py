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

def build_layer(state_before, flag_dropout, trng = None, default_seed = 123):
	trng = RandomStreams(default_seed)
	
	proj = T.switch(
			flag_dropout,
			(state_before
				* trng.binomial(
					state_before.shape,
					p = 0.5,
					n = 1,
					dtype = state_before.dtype
					)
			),
			state_before * 0.5
		)

	return proj

