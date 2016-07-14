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

import common
import rnn, gru, lstm

rnn_modules = {
		'rnn':rnn,
		'gru':gru,
		'lstm':lstm,
	}

def build(rnn_name, tparams, prefix, state_before, mask, dim, odim = None):
	if odim is None:
		odim = dim

	rnn_module = rnn_modules[rnn_name]

	proj_fw = rnn_module.build(tparams, '%s_fw'%(prefix), state_before, mask, dim, odim)
	proj_bw = rnn_module.build(tparams, '%s_bw'%(prefix), state_before[::-1], mask[::-1], dim, odim)[::-1]

	params = [
		('%s_Wf'%(prefix), common.ortho_weight(odim, odim)),
		('%s_Wb'%(prefix), common.ortho_weight(odim, odim)),
		('%s_b'%(prefix), common.rand_weight((odim,), floatX)),
		]
	
	for name, value in params:
		tparams[name] = theano.shared(value, name = name)

	proj = T.dot(proj_fw, tparams['%s_Wf'%(prefix)]) + T.dot(proj_bw, tparams['%s_Wb'%(prefix)]) + tparams['%s_b'%(prefix)]

	return proj

