#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.06
Description: run this script to run create a list of vocabulary for a dataset
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
from optparse import OptionParser

def dump_wemb(fname_vocab, fname_wemb, fname_output):
	vocabs = open(fname_vocab, 'r').decode('utf8').read().split('\n')
	vocabs = set(vocabs)
	n_vocab = len(vocabs)
	count = 0

	fobj_wemb = open(fname_wemb, 'r')	
	fobj_output = open(fname_output, 'w')

	for line in fobj_wemb:
		params = line.replace('\n', '').split(' ')
		if params[0] in vocabs:
			fobj_output.write(line)
			count += 1
			if count == n_vocab:
				break

	fobj_wemb.close()
	fobj_output.close()

def main():
	optparser = OptionParser()
	optparser.add_option('-v', '--vocab', action='store', dest='fname_vocab')
	optparser.add_option('-w', '--wemb', action='store', dest='fname_wemb')
	optparser.add_option('-o', '--output', action='store', dest='fname_output')
	opts, args = optparser.parse_args()

	dump_wemb(fname_vocab, fname_wemb, fname_output)	

if __name__ == '__main__':
	main()
