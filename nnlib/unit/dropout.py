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

def build_layer(state_before, flag, rate = 0.5):
	trng = MRG_RandomStreams(default_seed)

	proj = T.switch(
			flag,
			(state_before
				* trng.binomial(
					state_before.shape,
					p = rate,
					dtype = state_before.dtype
					)
			),
			state_before * rate
		)

	return proj

