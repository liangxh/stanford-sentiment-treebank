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

import nltk

def dump(fname, vocabs):
	try:
		fobj = open(fname, 'w')
		fobj.write('\n'.join(vocabs))
		fobj.close()
		return True
	except:
		return False

def build_vocab(ifname):
	dataset = cPickle.load(open(ifname, 'r'))

	vocabs = set()

	for xy in dataset:
		seqs, y = xy
		for seq in seqs:
			toks = nltk.word_tokenize(seq)
			vocabs |= set(toks)

	return list(vocabs)

def main():
	optparser = OptionParser()
	optparser.add_option('-d', '--fname_dataset', action='store', dest='fname_dataset', type='str')
	optparser.add_option('-v', '--fname_vocab', action='store', dest='fname_vocab', type='str')
	opts, args = optparser.parse_args()

	vocabs = build_vocab(opts.fname_dataset)
	print >> sys.stderr, 'main: [info] %d vocabs'%(len(vocabs))
	dump(opts.fname_vocab, vocabs)

if __name__ == '__main__':
	main()
