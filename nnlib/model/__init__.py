import lstm
import gru
import rnn
import brnn
import blstm
import grurnn
import lstm2l

models = {
	'lstm':lstm,
	'gru':gru,
	'rnn':rnn,
	'brnn':brnn,
	'blstm':blstm,
	'grurnn':grurnn,
	'lstm2l':lstm2l,
	}

def get(name):
	model = models.get(name, None)

	if model is not None:
		return model
	else:
		raise Warning('model %s not found'%(name))
