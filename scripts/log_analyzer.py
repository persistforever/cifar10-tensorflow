# -*- encoding: utf-8 -*-
# author: ronniecao
import os
import re
import matplotlib.pyplot as plt
import numpy


def load_log(path):
	with open(path, 'r') as fo:
		loss, loss_list, precision_list = [], [], []
		for line in fo:
			line = line.strip()
			pattern1 = re.compile(r'loss: ([\d\.]+)')
			pattern2 = re.compile(r'epoch: ([\d]+), valid precision: ([\d\.]+)')
			loss_res = pattern1.findall(line)
			precision_res = pattern2.findall(line)
			if loss_res:
				loss.append(float(loss_res[0]))
			if precision_res:
				precision_list.append(float(precision_res[0][1]))
				if loss:
					loss_list.append(numpy.array(loss).mean())
				loss = []
	return loss_list, precision_list

def curve_smooth(data_list, batch_size=100):
	new_data_list, idx_list = [], []
	for i in range(int(len(data_list) / batch_size)):
		batch = data_list[i*batch_size: (i+1)*batch_size]
		new_data_list.append(1.0 * sum(batch) / len(batch))
		idx_list.append(i*batch_size)

	return new_data_list, idx_list

def plot_curve(loss_list, loss_idxs, precision_list, precision_idxs):
	fig = plt.figure()
	plt.subplot(121)
	p1 = plt.plot(loss_idxs, loss_list, '.--', color='#6495ED')
	plt.grid(True)
	plt.title('cifar10 image classification loss')
	plt.xlabel('# of epoch')
	plt.ylabel('loss')
	plt.subplot(122)
	p2 = plt.plot(precision_idxs, precision_list, '.--', color='#66CDAA')
	plt.grid(True)
	plt.title('cifar10 image classification valid precision')
	plt.xlabel('# of epoch')
	plt.ylabel('accuracy')
	plt.show()


loss_list, precision_list = load_log('E:\\Github\cifar10-tensorflow\\results\cifar10-v3\cifar10-v3.txt')
print(len(loss_list), len(precision_list))
print(numpy.array(loss_list[900:1000]).mean(), numpy.array(precision_list[900:1000]).mean())
loss_list, loss_idxs = curve_smooth(loss_list, batch_size=1)
precision_list, precision_idxs = curve_smooth(precision_list, batch_size=1)
plot_curve(loss_list, loss_idxs, precision_list, precision_idxs)