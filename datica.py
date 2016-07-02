# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
dir_root = 'data/corpus/'

def read_text():
	ifname = dir_root + 'stanfordSentimentTreebank/SOStr.txt'
	lines = open(ifname, 'r').read().split('\n')

	texts = []
	for line in lines:
		params = line.split('|')
		if len(params) > 1:
			text = ' '.join(params)
			texts.append(text)

	return texts

def read_splitlabel():
	ifname = dir_root + 'stanfordSentimentTreebank/datasetSplit.txt'
	lines = open(ifname, 'r').read().split('\n')

	splitlabels = []
	for line in lines[1:]:
		params = line.split(',')
		if len(params) == 2:
			splitlabels.append(int(params[1]))
	
	return splitlabels

def read_sentiscore():
	ifname = dir_root + 'stanfordSentimentTreebank/sentiment_labels.txt'
	lines = open(ifname, 'r').read().split('\n')

	sentiscores = []
	for line in lines[1:]:
		params = line.split('|')
		if len(params) == 2:
			sentiscores.append(float(params[1]))

	return sentiscores

def read_phraseid():
	ifname = dir_root + 'stanfordSentimentTreebank/dictionary.txt'
	lines = open(ifname, 'r').read().split('\n')

	phraseid = {}
	for line in lines:
		params = line.split('|')
		if len(params) == 2:
			phraseid[params[0]] = int(params[1])

	return phraseid

def prepare():
	texts = read_text()
	splitlabels = read_splitlabel()
	sentiscores = read_sentiscore()
	phraseid = read_phraseid()

	train_text = []
	train_label = []
	
	valid_text = []
	valid_label = []

	test_text = []
	test_label = []

	n_sample = len(texts)
	if n_sample == len(splitlabels) and len(sentiscores) == len(phraseid):
		print '%d samples'%(n_sample)
	else:
		print 'reading fail'

	for i, didx in enumerate(splitlabels):
		if didx == 1:
			list_text = train_text
			list_label = train_label
		elif didx == 3:
			list_text = valid_text
			list_label = valid_label
		elif didx == 2:
			list_text = test_text
			list_label = test_label

		list_text.append(texts[i])
		list_label.append(sentiscores[phraseid[texts[i]]])

	dataset = ((train_text, train_label), (valid_text, valid_label), (test_text, test_label))
	cPickle.dump(dataset, open('data/raw.pkl', 'w'))

def prepare_finegrain():
	dataset = cPickle.load(open('data/raw.pkl', 'r'))
	
	def labelize(x_label):
		x, label = x_label
		y = []
		for l in label:
			if l <= 0.2:
				y.append(0)
			elif l <= 0.4:
				y.append(1)
			elif l <= 0.6:
				y.append(2)
			elif l <= 0.8:
				y.append(3)
			else:
				y.append(4)

		print len(y)
		return (x, y)

	train, valid, test = dataset
	cPickle.dump((labelize(train), labelize(valid), labelize(test)), open('data/finegrain.pkl', 'w'))

def prepare_binary():
	dataset = cPickle.load(open('data/raw.pkl', 'r'))
	
	def labelize(x_label):
		x, label = x_label
		y = []
		for l in label:
			if l <= 0.4:
				y.append(0)
			elif l > 0.6:
				y.append(1)

		print len(y)
		return (x, y)

	train, valid, test = dataset
	cPickle.dump((labelize(train), labelize(valid), labelize(test)), open('data/binary.pkl', 'w'))

def main():
	prepare()
	prepare_finegrain()
	prepare_binary()

if __name__ == '__main__':
	main()
