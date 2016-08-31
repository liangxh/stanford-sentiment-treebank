#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.30
'''

import theano
import theano.tensor as T
from collections import OrderedDict

from flag import Flag

class Model:
	def __init__(self):
		self.tparams = OrderedDict()
		self.flag_training = Flag(False)

	def build(self, options):
		raise NotImplementedError

	def load_value(self, name, value):
		self.tparams[name].set_value(value)

	def get_params(self):
		'''
		get value from tparams to an OrderedDict for saving
		'''
		params = OrderedDict()
		for name, value in self.tparams.iteritems():
			params[name] = value.get_value()

		return params

	def update_tparams(self, params):
		'''
		set values for tparams using params
		'''
		for name, value in self.tparams.iteritems():
			if not name in params:
				raise Warning('param %s not found'%(name))

			self.tparams[name].set_value(params[name])

	def predict(self, *inputs):
		return self.f_pred(*inputs)

	def generate(self, inputs, output, pred, cost, optimizer):
		
		if not isinstance(inputs, list):
			inputs = [inputs, ]

		outputs = [output, ] # for possible extension

		self.f_pred = theano.function(inputs, pred, name='f_pred_proba')
		
		#self.f_pred_proba = theano.function(inputs, pred, name='f_pred_proba')
		#self.f_pred = theano.function(inputs, pred.argmax(axis=1), name='f_pred')
		
		self.f_cost = theano.function(inputs + outputs, cost, name = 'f_cost')
		grads = T.grad(cost, wrt = self.tparams.values())

		self.learning_rate = T.scalar(name = 'lr')
		self.f_grad_shared, self.f_update = optimizer.build(
								self.learning_rate, self.tparams,
								grads, inputs, output, cost
							)

	def train(self, inputs, output, learning_rate):
		if isinstance(inputs, tuple):
			inputs = list(inputs)

		if not isinstance(inputs, list):
			inputs = [inputs, ]
		
		outputs = [output, ]  # for possible extension

		self.flag_training.on()

		cost = self.f_grad_shared(*(inputs + outputs))
		self.f_update(learning_rate)

		self.flag_training.off()
	
		return cost


