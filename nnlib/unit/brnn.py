#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.09
Description: a factory for building the Bidirectional Recurrent Neural Network layer
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

import rnn, linear_combo
from nnlib.common import State

def build(rnn_module, tparams, prefix, state_before, mask, odim = None,
		bias = False,
		post_process = False,
	):
	if odim is None:
		odim = state_before.odim

	state_fw = rnn_module.build(tparams, '%s_fw'%(prefix), state_before, mask, odim)

	state_bw = rnn_module.build(tparams, '%s_bw'%(prefix),
				state_before.map(lambda k:k[::-1]), mask[::-1], odim
			).map(lambda k:k[::-1])

	if post_process:
		state_fw = rnn.postprocess_avg(state_fw, mask)
		state_bw = rnn.postprocess_avg(state_bw, mask)

	state = linear_combo.build(
			tparams, '%s_lr'%(prefix), [state_fw, state_bw], odim, bias)

	return state

