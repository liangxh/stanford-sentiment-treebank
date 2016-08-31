import numpy as np
import collections

DataPair = collections.namedtuple('DataPair', ['x', 'y'])

def batch_iterate(n, batch_size, shuffle = False):
	idx_list = np.arange(n)
		
	if shuffle:
		np.random.shuffle(idx_list)

	batch_start = 0
	for i in range(n // batch_size):
		yield idx_list[batch_start:(batch_start + batch_size)]
		batch_start += batch_size

	if batch_start != n:
		yield idx_list[batch_start:]		

class BatchIterator:
	def __init__(self, data):
		self._data = data
		self.num = len(data)

	def iterate(self, batch_size, shuffle = False):
		for idx_batch in batch_iterate(self.num, batch_size, shuffle):
			yield [self._data[idx] for idx in idx_batch]

	def __getitem__(self, idx):
		return self._data[idx]

class DataPairBatchIterator(BatchIterator):
	def iterate(self, batch_size, shuffle = False):
		for idx_batch in batch_iterate(self.num, batch_size, shuffle):
			x = []
			y = []

			for idx in idx_batch:	
				sample = self._data[idx]
				x.append(sample.x)
				y.append(sample.y)
			
			yield (x, y)

	def x(self):
		return map(lambda k:k.x, self._data)
	
	def y(self):
		return map(lambda k:k.y, self._data)

class DataSet:
	def __init__(self, train, valid, test):
		self.ydim = np.max(map(lambda k:k.y, test)) + 1

		self.train = DataPairBatchIterator(train)
		self.valid = DataPairBatchIterator(valid)
		self.test = DataPairBatchIterator(test)

