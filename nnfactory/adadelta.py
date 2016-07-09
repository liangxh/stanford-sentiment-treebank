#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.09
Description: an adaptive learning rate optimizer, implemented based on open 
'''

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX

def build(lr, tparams, grads, inputs, y, cost):
	grad_inputs = []
	grad_inputs.extend(inputs)
	grad_inputs.append(y)

	zipped_grads = [theano.shared(p.get_value() * np.asarray(0., dtype = floatX), name = '%s_grad' % k)
					for k, p in tparams.iteritems()]
	running_up2 = [theano.shared(p.get_value() * np.asarray(0., dtype = floatX), name = '%s_rup2' % k)
					for k, p in tparams.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * np.asarray(0., dtype = floatX), name = '%s_rgrad2' % k)
					for k, p in tparams.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
			 for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function(
				grad_inputs, cost,
				updates = zgup + rg2up,
				name = 'adadelta_f_grad_shared'
			)

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
			for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
			for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

	f_update = theano.function(
				[lr], [],
				updates = ru2up + param_up,
				on_unused_input = 'ignore',
				name = 'adadelta_f_update'
			)

	return f_grad_shared, f_update

