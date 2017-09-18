from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

import tensorflow as tf
import numpy as np

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, n_nuerons=100, optimizer_class=tf.train.AdamOptimizer, 
            learning_rate=0.01, batch_size=20, activation=tf.nn.elu, initializer=he_init, batch_norm_momentum=None,
            dropout_rate=None, random_state=None):
        self.n_hidden_layers = n_hidden_layers
        self.n_nuerons = n_nuerons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        """Build the hidden layers, with support for batch normalization and dropout."""
        for layer in range(self.n_hidden_layers):
            if (self.dropout_rate):
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.n_nuerons, kernel_initializer=self.initializer, name="hidden%d" % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum, training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer + 1))  
        return inputs
        
