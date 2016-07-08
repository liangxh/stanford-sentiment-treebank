#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.07.06
Description: WordEmbedder is a module for reading embedding vectors and embedding sequence of tokens
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import theano

class WordEmbedder:
	def __main__(self, Widx, Wemb, dim):
		self.Widx = Widx
		self.Wemb = Wemb
		self.dim = dim

	def get_Wemb(self):
		return np.asarray(Wemb).astype(theano.config.floatX)

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
			vec = [float(p) for p in params[1:]]
			
			Widx[word] = idx
			idx += 1
			Wemb.append(vec)

		dim = len(Wemb[0])

		return WordEmedder(Widx, Wemb, dim)

def test():
	fname_embed = 'data/wemb/binary.glove.6B.50d.txt'
	wembedder = WordEmbedder.load()

	sample = 'hello how are you ?'.split(' ')
	print >> sys.stderr, wembedder.seq2idx(sample)

if __name__ == '__main__':
	test()
