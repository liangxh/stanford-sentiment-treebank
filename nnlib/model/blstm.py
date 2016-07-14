from .. import unit, optimizer

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX
#theano.config.exception_verbosity = 'high'

def build(options):
	tparams = OrderedDict()

	# global flag for dropout
	flag_dropout = theano.shared(np.asarray(0., dtype = floatX))

	# a matrix, whose shape is (n_timestep, n_samples) for theano.scan in training process
	x = T.matrix('x', dtype = 'int64')
		
	# a matrix, used to distinguish the valid elements in x
	mask = T.matrix('mask', dtype = floatX)

	# a vector of targets for $n_samples samples   
	y = T.vector('y', dtype = 'int64')

	n_timesteps = x.shape[0]
	n_samples = x.shape[1]

	tparams['Wemb'] = theano.shared(
				np.zeros((options['vocab_size'], options['dim_wemb']), dtype = floatX),
				name = 'Wemb'
			)

	# transfer x, the matrix of tids, into Wemb, the 'matrix' of embedding vectors 
	emb = tparams['Wemb'][x.flatten()].reshape(
			[n_timesteps, n_samples, options['dim_wemb']]
		)

	# the result of LSTM, a matrix of shape (n_timestep, n_samples, dim_hidden)
	proj = unit.brnn.build('lstm', tparams, 'blstm', emb, mask, options['dim_wemb'], options['dim_hidden'])
	proj = unit.rnn.postprocess_avg(proj, mask)

	proj = unit.dropout.build_layer(proj, flag_dropout)
	pred = unit.softmax.build_layer(tparams, 'softmax', proj, options['dim_hidden'], options['ydim'])

	# build functions for prediction
	f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
	f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

	# build function for cost calculation
	off = 1e-8
	if pred.dtype == 'float16':
		off = 1e-6

	cost = -T.log(pred[T.arange(n_samples), y] + off).mean()

	if options['decay_c'] > 0.:
		decay_c = theano.shared(np.asarray(options['decay_c'], dtype = floatX), name='decay_c')
		weight_decay = 0.
		weight_decay += (tparams['softmax_U'] ** 2).sum()
		weight_decay *= decay_c
		cost += weight_decay
	
	f_cost = theano.function([x, mask, y], cost, name = 'f_cost')
		
	grads = T.grad(cost, wrt = tparams.values())
	#f_grad = theano.function([x, mask, y], grads, name = 'f_grad')

	lr = T.scalar(name = 'lr')
	f_grad_shared, f_update = optimizer.adadelta.build(lr, tparams, grads, [x, mask], y, cost)

	return tparams, f_pred, f_pred_prob, f_grad_shared, f_update, flag_dropout

