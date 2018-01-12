# dnn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

pwd = "/home/enningxie/Documents/DataSets/mnist/"
learning_rate = 0.05
layer1 = 784
layer2 = 128
layer3 = 10

class DNN:
    def __init__(self):
        pass


def load_mnist():
    data, label = input_data.read_data_sets(pwd)
    return data, label


def main(_):
    print("x")
    data, label = load_mnist()
    print(data.shape)
    print(label.shape)


if __name__ == "__main__":
    tf.app.run()