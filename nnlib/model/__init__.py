import importlib

_prefix = 'nnlib.model.'

def get(name):
	try:
		model = __import__(_prefix + name, fromlist=[''])
		return model
	except:
		raise Warning('model %s not found'%(name))
		
