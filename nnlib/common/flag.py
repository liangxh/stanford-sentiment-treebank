#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.30
Description: a module which builds flag, for dropout so far  
'''

import numpy as np
import theano
floatX = theano.config.floatX

class Flag:
	def __init__(self, init_value = False):
		if isinstance(init_value, bool):
			init_value = 1. if init_value else 0.
		elif not init_value == 1. and not init_value == 0.:
			raise Warning('invalue initial value for Flag')

		self.var = theano.shared(np.asarray(init_value, dtype = floatX))

	def on(self):
		self.var.set_value(1.)

	def off(self):
		self.var.set_value(0.)

