import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
import matplotlib.pyplot as plt

# mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist/", one_hot=True)

# print(tf.stack([mnist.train.images[1]]).shape)

# r = cv2.imread("/home/enningxie/Documents/DataSets/8.jpg")
# gray_r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
# print(gray_r/255)
# cv2.imshow('opencv', gray_r)
# cv2.waitKey()
#
# arr = []
#
# for i in range(28):
#     for j in range(28):
#         # mnist 里的颜色是0代表白色（背景），1.0代表黑色
#         pixel = 1.0 - float(gray_r[i, j])/255.0
#         # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
#         arr.append(pixel)
#
# arr2 = np.asarray(arr)
# im = arr2.reshape([28, 28])
# cv2.imshow('opencv', im)
# cv2.waitKey()
# # print(arr2.shape)
# # print('----------------------')
# # print(arr2)

###################################

mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist/", one_hot=True)

for i in range(10):
    plt.imshow()