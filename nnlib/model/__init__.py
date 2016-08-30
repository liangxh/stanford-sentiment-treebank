import importlib

_prefix = 'nnlib.model.'

def get(module_name, class_name = None):
	if class_name is None:
		class_name = module_name

	class_model = __import__(_prefix + module_name, fromlist=['']).__dict__[class_name]
	return class_model()

