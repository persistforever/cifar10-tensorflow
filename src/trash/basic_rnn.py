# -*- encoding: utf8 -*-
import numpy
import pandas
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class Dataset:
    
    def __init__(self, path):
        # read iris_training set and test set
        dataset = pandas.read_csv(path)
        self.flower, self.label = dataset.iloc[:,:-1].values, dataset.iloc[:,-1].values
        self.num_examples = self.flower.shape[0]
        self.index = 0
        
    def next_batch(self, batch_size):
        batch_flower = self.flower[self.index:self.index+batch_size,:]
        batch_label = self.label[self.index:self.index+batch_size]
        self.index += batch_size
        if self.index >= self.num_examples:
            self.index = 0
        return [batch_flower, batch_label]


class RNN:
    
    def construct_model(self, batch_size):
        # input variable
        self.flower = tf.placeholder(dtype=tf.float32, shape=[batch_size, 3], name='flower')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        flower_series = tf.unstack(self.flower, num=3, axis=1, name='flower_series')
        label = tf.one_hot(self.label, depth=3, dtype=tf.float32, name='label')
        
        # recurrent layer
        # init state
        init_state = tf.Variable(
            initial_value=tf.zeros(shape=[batch_size, 16], dtype=tf.float32),
            name='init_state')
        # filter
        W_rnn = tf.Variable(
            initial_value=tf.random_normal(shape=[17, 16], mean=0.0, stddev=0.01), 
            name='W_rnn')
        # bias
        b_rnn = tf.Variable(
            initial_value=tf.zeros(shape=[16]), 
            name='b_rnn')
        # hidden states
        current_state = init_state
        state_series = []
        for current_input in flower_series:
            current_input = tf.reshape(current_input, shape=[batch_size, 1])
            concat_vector = tf.concat([current_input, current_state], axis=1)
            next_state = tf.tanh(tf.matmul(concat_vector, W_rnn) + b_rnn)
            self.next_state = next_state
            state_series.append(next_state)
            current_state = next_state
        
        # softmax layer
        # weight
        W_softmax = tf.Variable(
            initial_value=tf.random_normal(shape=[16, 3], mean=0.0, stddev=0.01), 
            name='W_softmax')
        # bias
        b_softmax = tf.Variable(
            initial_value=tf.zeros(shape=[3]), 
            name='b_softmax')
        # softmax
        label_pred = tf.nn.softmax(
            logits=tf.matmul(state_series[-1], W_softmax) + b_softmax)
        
        # objective function and optimizer
        self.objective = - tf.reduce_sum(label * tf.log(label_pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.objective)
        # observation
        correct_prediction = tf.equal(tf.argmax(label_pred, 1), tf.argmax(label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
    def train(self, trainset, batch_size=128, epochs=20):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch in range(int(trainset.num_examples / batch_size)):
                batch_flower, batch_label = trainset.next_batch(batch_size)
                sess.run(self.optimizer, feed_dict={
                    self.flower:batch_flower, self.label:batch_label})
            print(('epoch:%i, accuracy: %.4f') % 
                  (epoch, sess.run(self.accuracy,
                                   feed_dict={self.flower:batch_flower, 
                                              self.label:batch_label})))
            
    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print(numpy.array([[1, 2, 3], [4, 5, 6]]))
        temp = sess.run(self.next_state, feed_dict={
            self.flower:numpy.array([[1, 2, 3], [4, 5, 6]])})
        print(temp.shape)


trainset = Dataset('../data/iris/iris_training.csv')
testset = Dataset('../data/iris/iris_test.csv')
rnn = RNN()
rnn.construct_model(batch_size=2)
rnn.train(trainset, batch_size=10)
# rnn.debug()