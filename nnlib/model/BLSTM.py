from .. import unit, optimizer

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX
#theano.config.exception_verbosity = 'high'

from nnlib.common import State, Flag, Model

class BLSTM(Model):
	def build(self, options):
		x = T.matrix('x', dtype = 'int64')
		
		mask = T.matrix('mask', dtype = floatX)

		y = T.vector('y', dtype = 'int64')

		batch_size = x.shape[1]

		state = unit.embedding.build(
				self.tparams, 'Wemb', x,
				options['vocab_size'], options['dim_wemb']
			)

		state = unit.brnn.build(
					unit.lstm,
					self.tparams, 'blstm', state, mask, options['dim_hidden'],
					bias = True,
					post_process = True,
					)

		state = unit.dropout.build(state, self.flag_training.var)
		state = unit.linear_combo.build(self.tparams, 'lr', state, options['ydim'], bias = True)
		pred = T.nnet.softmax(state.var)

		off = 1e-6 if pred.dtype == 'float16' else 1e-6

		cost = -T.log(pred[T.arange(batch_size), y] + off).mean()

		if options['decay_c'] > 0.:
			decay_c = theano.shared(np.asarray(options['decay_c'], dtype = floatX), name='decay_c')
			weight_decay = 0.
			weight_decay += (self.tparams['lr_W'] ** 2).sum()
			weight_decay *= decay_c
			cost += weight_decay

		self.generate([x, mask], y, pred, cost, optimizer.adadelta)

