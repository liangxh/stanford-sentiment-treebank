#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.30
Description: a module for transfering states through the neural network flow
'''

class State:
	def __init__(self, var, odim):
		self.var = var      	# variable
		self.odim = odim	# last dim of self.var

	def map(self, func):
		self.var = func(self.var)
		return self
