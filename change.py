
import numpy as np
import sys
import commands
from collections import OrderedDict


def main():
	fname = sys.argv[1]
	key_old = sys.argv[2]
	key_new = sys.argv[3]

	params = np.load(fname)
	new_params = OrderedDict()

	flag = False
	for k, v in params.iteritems():
		if k == key_old:
			k = key_new
			flag = True

		new_params[k] = v

	if flag:
		print '[info] %s found in %s'%(key_old, fname)
		commands.getoutput('rm -f %s'%(fname))
		np.savez(fname, **new_params)
	else:
		print '[info] %s not found in %s'%(key_old, fname)
	

if __name__ == '__main__':
	main()
