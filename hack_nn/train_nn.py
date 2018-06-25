import struct
from hack_nn import *
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    # train_data_set, train_labels = get_training_data_set()
    # test_data_set, test_labels = get_test_data_set()
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    network = Network([784, 300, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_data, 0.3, 1)
        print('epoch %d finished' % (epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data, test_labels)
            print('after epoch %d, error ratio is %f' % (epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


if __name__ == '__main__':
    train_and_evaluate()
