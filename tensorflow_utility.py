# coding: utf-8
import tensorflow as tf


#
# 初期化された重みを返す
#
# @param shape
#
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


#
# 初期化されたバイアスVariableを返す
#
# @param shape
#
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


#
# 畳み込みを行い特徴マップを返す
#
# @param x
# @param W
#
def conv2d(x, w):
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    return tf.nn.conv2d(x, w, strides=strides, padding=padding)


#
# 最大値プーリングの結果を返す
#
# @param x
#
def max_pool_2x2(x):
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'SAME'
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)
