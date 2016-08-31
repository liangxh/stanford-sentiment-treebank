#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.08.30
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import nltk
import cPickle
from optparse import OptionParser

import nnlib
from nnlib.app import Trainer
from nnlib.common.dataset import DataSet, DataPair

from wordembedder import WordEmbedder

def main():
	optparser = OptionParser()

	optparser.add_option('-p', '--prefix', action='store', dest='prefix')
	optparser.add_option('-i', '--input', action='store', dest='key_input')
	optparser.add_option('-e', '--embed', action='store', dest='key_embed')
	optparser.add_option('-m', '--model', action='store', dest='model_name')
	optparser.add_option('-r', '--resume', action='store_true', dest='resume', default = False)

	optparser.add_option('-d', '--dim_hidden', action='store', dest='dim_hidden', type='int', default = None)
	optparser.add_option('-b', '--batch_size', action='store', type='int', dest='batch_size', default = 16)
	optparser.add_option('-l', '--learning_rate', action='store', type='float', dest='learning_rate', default = 0.05)
	optparser.add_option('-c', '--decay_c', action='store', type='float', dest='decay_c', default = 1e-4)
	
	opts, args = optparser.parse_args()


	prefix = opts.prefix
	fname_input = 'data/dataset/' + '%s.pkl'%(opts.key_input)
	fname_embed = 'data/wemb/' + '%s.txt'%(opts.key_embed)

	fname_model = 'data/model/' + '%s'%(prefix)
	
	dataset = cPickle.load(open(fname_input, 'r'))
	wembedder = WordEmbedder.load(fname_embed)

	def preprocess_text(wembedder, xy):
		texts, y = xy
		seqs = [nltk.word_tokenize(t.lower()) for t in texts]
		idxs = [wembedder.seq2idx(seq) for seq in seqs]

		return (idxs, y)
	
	dataset = [preprocess_text(wembedder, subset) for subset in dataset]

	dataset = DataSet(*[
			[DataPair(x, y)  for x, y in zip(*subset)]
				for subset in dataset])

	Wemb = wembedder.get_Wemb() if not opts.resume else None

	trainer = Trainer()

	print >> sys.stderr, 'main: [info] start training'
	res = trainer.train(
			dataset = dataset,
			Wemb = Wemb,

			fname_model = fname_model,
			resume = opts.resume,

			model_name = opts.model_name,
			dim_hidden = opts.dim_hidden,
			batch_size = opts.batch_size,
			decay_c = opts.decay_c,
			learning_rate = opts.learning_rate,
		)

if __name__ == '__main__':
	main()
