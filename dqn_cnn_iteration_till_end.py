"""DQN Class

DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""
import numpy as np
import tensorflow as tf
import gzip
import collections
import time
import math as m
import scipy.io as sio
import os


class DQN:

    def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str="main") -> None:
        """DQN Agent can

        1) Build network
        2) Predict Q_value given state
        3) Train parameters

        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()


    def conv_start(x, stride, W, b):
        x = tf.reshape(x, [-1, 9, 9, 1])
        layer_1 = tf.add(tf.nn.conv2d(x, W, strides=stride, padding='SAME'), b)
        return layer_1


    def conv_one(x, stride, W, b):
        layer_1 = tf.add(tf.nn.conv2d(x, W, strides=stride, padding='SAME'), b)
        return layer_1


    def conv_end(x, stride, W, b):
        layer_1 = tf.nn.conv2d(x, W, strides=stride, padding='SAME') + b
        layer_1 = tf.reshape(layer_1, [-1, 1, 1, 3*3*64])
        return layer_1


    def relu(x):
        layer_1 = tf.nn.relu(x)
        return layer_1




    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)



    def _build_network(self, h_size=128, l_rate=0.00001) -> None:
            """DQN Network architecture (simple MLP)

            Args:
                h_size (int, optional): Hidden layer dimension
                l_rate (float, optional): Learning rate
            """
            Wc_1 = DQN.weight_variable([4,4,1,32])
            bc_1 = tf.Variable(tf.random_normal([32]))

            Wc_2 = DQN.weight_variable([3,3,32,64])
            bc_2 = tf.Variable(tf.random_normal([64]))

            Wc_3 = DQN.weight_variable([3,3,64,64])
            bc_3 = tf.Variable(tf.random_normal([64]))

            #Wc_4 = weight_variable([3,1,64,64])
            #bc_4 = tf.Variable(tf.random_normal([64]))

            #Wc_e = weight_variable([3,3,64,1])
            #bc_e = tf.Variable(tf.random_normal([1]))
            with tf.variable_scope(self.net_name):
                #input_data = tf.placeholder("float", shape=[None, n_input], name='Input_data')
                self.input_data = tf.placeholder("float", shape=[None,self.input_size*self.input_size], name='Input_data')
                net = self.input_data
                net = DQN.conv_start(net, [1,2,2,1], Wc_1, bc_1)
                net = DQN.relu(net)
                net = DQN.conv_one(net, [1,2,2,1], Wc_2, bc_2)
                net = DQN.relu(net)
                net = DQN.conv_end(net, [1,1,1,1], Wc_3, bc_3)
                net = DQN.relu(net)
                #self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
                #net = self._X

                net = tf.layers.dense(net, h_size)
                net = DQN.relu(net)
                net = tf.layers.dense(net, self.output_size)
                #saver = tf.train.Saver()
                self._Qpred = net

                self._Y = tf.placeholder(tf.float32, shape=[None,1,1,self.output_size])
                self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

                #optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
                optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate,momentum=0.9)
                self._train = optimizer.minimize(self._loss)



    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)

        Args:
            state (np.ndarray): State array, shape (n, input_dim)

        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """

        #x = np.reshape(state, [-1, self.input_size,])


        #x = np.reshape(state, [1,self.input_size,self.input_size,1])
        return self.session.run(self._Qpred, feed_dict={self.input_data: state})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result

        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)

        Returns:
            list: First element is loss, second element is a result from train step
        """
        feed = {
            self.input_data: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)
