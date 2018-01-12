# about tf.estimator
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from urllib.request import urlopen

import tensorflow as tf
import numpy as np

IRIS_TRAINING = '/home/enningxie/Documents/DataSets/iris/iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = '/home/enningxie/Documents/DataSets/iris/iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'


# download and store the dataset
def download(pwd, url):
    if not os.path.exists(pwd):
        print("Downloading Training Set.")
        raw = urlopen(url).read()
        with open(IRIS_TRAINING, 'wb') as f:
            f.write(raw)
    else:
        print("existed.")


# load dataset
def load(pwd):
    data_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=pwd,
        target_dtype=np.int,
        features_dtype=np.float32
    )
    return data_set


# construct a dnn classifier
def model():
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    # build 3 layer dnn with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/home/enningxie/Documents/Models/"
                                            )
    return classifier


# train
def train(model, input_data):
    # input
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(input_data.data)},
        y=np.array(input_data.target),
        num_epochs=None,
        shuffle=True
    )
    model.train(input_fn=train_input, steps=2000)


# test
def test(model, input_data):
    test_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(input_data.data)},
        y=np.array(input_data.target),
        num_epochs=1,
        shuffle=False
    )
    # evaluate accuracy.
    accuracy_score = model.evaluate(input_fn=test_input)['accuracy']
    print("Test set accuracy: ", accuracy_score)


# predict
def predict(model, input_data):
    predict_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": input_data},
        num_epochs=1,
        shuffle=False
    )
    predictions = list(model.predict(input_fn=predict_input))
    predicted_classes = [p['classes'] for p in predictions]
    print("New samples, class predictions: ", predicted_classes)


# main
def main():
    # download dataset
    download(IRIS_TRAINING, IRIS_TRAINING_URL)
    download(IRIS_TEST, IRIS_TEST_URL)

    # load dataset
    training_set = load(IRIS_TRAINING)
    test_set = load(IRIS_TEST)

    classifier = model()
    # train
    train(classifier, training_set)
    # test
    test(classifier, test_set)
    # predict
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict(classifier, new_samples)

    print("end.")


if __name__ == "__main__":
    main()
