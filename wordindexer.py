#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.06
Description: WordEmbedder is a module for transferring sequence of words to indexes
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import theano

class WordIndexer:
	def __init__(self, Widx):
		self.Widx = Widx

	def seq2idx(self, seq):
		return [self.Widx[t] for t in seq if self.Widx.has_key(t)]

	@classmethod
	def load(self, fname_wemb):
		Widx = {}
		Wemb = []
	
		idx = 0

		fobj = open(fname_wemb, 'r')
		for line in fobj:
			params = line.replace('\n', '').split(' ')
			if len(params) < 2:
				continue
			
			word = params[0].decode('utf8')
			
			Widx[word] = idx
			idx += 1

		return WordIndexer(Widx)

def test():
	fname_embed = 'data/wemb/binary.glove.6B.50d.txt'
	windexer = WordIndexer.load(fname_embed)

	sample = 'hello how are you ?'.split(' ')
	print >> sys.stderr, windexer.seq2idx(sample)

if __name__ == '__main__':
	test()
