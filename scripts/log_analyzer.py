# -*- encoding: utf-8 -*-
# author: ronniecao
import os
import re
import matplotlib.pyplot as plt
import numpy


def load_log(path):
	with open(path, 'r') as fo:
		loss_list, train_precision_list, valid_precision_list = [], [], []
		for line in fo:
			line = line.strip()
			pattern = re.compile(r'epoch: ([\d]+), train precision: ([\d\.]+), train loss: ([\d\.]+), valid precision: ([\d\.]+)')
			res = pattern.findall(line)
			if res:
				loss_list.append(float(res[0][2]))
				train_precision_list.append(float(res[0][1]))
				valid_precision_list.append(float(res[0][3]))
	return loss_list, train_precision_list, valid_precision_list

def curve_smooth(data_list, batch_size=100):
	new_data_list, idx_list = [], []
	for i in range(int(len(data_list) / batch_size)):
		batch = data_list[i*batch_size: (i+1)*batch_size]
		new_data_list.append(1.0 * sum(batch) / len(batch))
		idx_list.append(i*batch_size)

	return new_data_list, idx_list

def plot_curve(loss_list, loss_idxs, train_precision_list, train_precision_idxs, valid_precision_list, valid_precision_idxs):
	fig = plt.figure()
	plt.subplot(121)
	p1 = plt.plot(loss_idxs, loss_list, '.--', color='#6495ED')
	plt.grid(True)
	plt.title('cifar10 image classification loss')
	plt.xlabel('# of epoch')
	plt.ylabel('loss')
	plt.subplot(122)
	p2 = plt.plot(train_precision_idxs, train_precision_list, '.--', color='#66CDAA')
	p3 = plt.plot(valid_precision_idxs, valid_precision_list, '.--', color='#FF6347')
	plt.legend((p2[0], p3[0]), ('train_precision', 'valid_precision'))
	plt.grid(True)
	plt.title('cifar10 image classification precision')
	plt.xlabel('# of epoch')
	plt.ylabel('accuracy')
	plt.show()
	# plt.savefig('E:\\Github\cifar10-tensorflow\\results\cifar10-v1\cifar10-v1.png', dpi=120, format='png')


loss_list, train_precision_list, valid_precision_list = load_log('E:\\Github\cifar10-tensorflow\\results\cifar10-v1\cifar10-v1.txt')
print(numpy.array(loss_list[-100:]).mean(), numpy.array(train_precision_list[-100:]).mean())
loss_list, loss_idxs = curve_smooth(loss_list[0:500], batch_size=1)
train_precision_list, train_precision_idxs = curve_smooth(train_precision_list, batch_size=10)
valid_precision_list, valid_precision_idxs = curve_smooth(valid_precision_list, batch_size=10)
plot_curve(loss_list, loss_idxs, train_precision_list, train_precision_idxs, valid_precision_list, valid_precision_idxs)