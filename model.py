import numpy as np
from layers.denseLayer import Dense
from layers.batchNormalization import BatchNorm
from typing import Literal
from losses import binary_cross_entropy, category_cross_entropy
from matplotlib import pyplot as plt
import tensorflow as tf

class NeuralNetwork:
    def __init__(self):
        # X input train data conating train data
        # Y input train label data for X input
        # layers_dims -- python list, containing the size and activation of each layer
        # optimizer - gd, momentum or adam opimizer
        # learning_rate -- the learning rate, scalar.
        # batch_size -- the size of a mini batch
        # epsilon -- hyperparameter preventing division by zero in any method
        # epochs -- number of epochs
        # cost - cost type. Here is categorical or binary
        # beta -- Momentum optimizer hyperparameter
        # beta1 -- Exponential decay hyperparameter for the past gradients estimates
        # beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
        # lambd -- regularization hyperparameter, scalar
       
        self.layers = []
      
    
    def Sequential(self, layers, batch_size=64):
      self.layers = layers
      self.batch_size = batch_size
      self._init_params()

    def _init_params(self):
      prev_shape = tf.reduce_prod(self.x_shape).numpy()
      for layer in self.layers:
        if hasattr(layer, 'init_params'):
          layer.init_params(prev_shape,self.optimizer)
          layer.init_batch_params(self.batch_size, prev_shape)
          prev_shape = layer.units
        else:
          layer.init_batch_params(self.batch_size, prev_shape)

    def assemble(self,input_shape, label_shape,
        optimizer: Literal["gd", "momentum", "adam"] = "gd",
         learning_rate0=0.1,
         epsilon=1e-10
         ,decay_steps=1000
         ,decay_rate=1):

        
        self.m = input_shape[0]
        self.y_shape = label_shape
        self.x_shape = input_shape[1]
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.learning_rate0 = learning_rate0
        self.learning_rate = tf.Variable(learning_rate0, dtype=tf.float32)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.t = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    
    def fit(self, dataset, epoch=2000):
      m = self.m
      self.costs = []

      padding_values = (tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.int64))
      mini_batches = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=([*self.x_shape], [self.y_shape]), padding_values=padding_values, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

      for i in range(epoch):
        self.cost_total = 0.0
        for minibatch_X, minibatch_Y in mini_batches:
          minibatch_X = tf.reshape(minibatch_X, shape=[minibatch_X.shape[0], -1])
          minibatch_X = tf.cast(minibatch_X, tf.float32)
          minibatch_Y = tf.cast(minibatch_Y, tf.float32)

          Al = self.forward(tf.transpose(minibatch_X))

          cost = self.compute_cost(Al, tf.transpose(minibatch_Y))
          
          self.cost_total = tf.add(tf.cast(cost, dtype=tf.float32), self.cost_total)

          # # back propagation
          self.backward(tf.transpose(minibatch_X),tf.transpose(minibatch_Y))

          # update paramters
          self.update_params(self.t)
          self.t.assign_add(1.0)

        # Print the cost every 1000 epoch
        tf.print("Cost after epoch", i+1, ":", self.cost_total / (m))
        self.costs.append(self.cost_total / m)

        # decay rate
        self.learning_rate.assign(self.learning_rate0 / (
            1 + (self.decay_rate * np.floor((i+1) / self.decay_steps))
        ))

      self.__plot()
    
    def fit_tf(self, dataset, epoch=500):
      m = self.m
      self.costs = []

      if self.optimizer == 'momentum':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lambda: self.learning_rate, momentum=0.9, nesterov=True)
      elif self.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lambda: self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
      else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lambda: self.learning_rate)

      padding_values = (tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.int64))
      mini_batches = dataset.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
      
      for i in range(epoch):
        self.cost_total = 0.0
        for minibatch_X, minibatch_Y in mini_batches:
          minibatch_X = tf.reshape(minibatch_X, shape=[minibatch_X.shape[0], -1])
          minibatch_X = tf.cast(minibatch_X, tf.float32)
          minibatch_Y = tf.cast(minibatch_Y, tf.float32)

          with tf.GradientTape() as t:
            Al = self.forward(tf.transpose(minibatch_X))

            cost = self.compute_cost(Al, tf.transpose(minibatch_Y))

          # # back propagation
          trainable_variables = [l_var for l in self.layers for l_var in l.variables]
          grads = t.gradient(cost, trainable_variables)

          # update paramters
          # optimizer.learning_rate.assign(self.learning_rate.numpy())
          optimizer.apply_gradients(zip(grads,trainable_variables))

          self.cost_total = tf.add(tf.cast(cost, dtype=tf.float32), self.cost_total)

        # Print the cost every 1000 epoch
        tf.print("Cost after epoch", i+1, ":", self.cost_total / (m))
        self.costs.append(self.cost_total / m)

        # decay rate
        self.learning_rate.assign(self.learning_rate0 / (
            1 + (self.decay_rate * np.floor((i+1) / self.decay_steps))
        ))

      self.__plot()


    def __plot(self):
      plt.figure(figsize=(8, 6))
      plt.plot([tensor.numpy() for tensor in self.costs])
      plt.ylabel("cost")
      plt.xlabel("epochs (per 100)")
      plt.title("Learning rate = " + str(self.learning_rate))
      plt.show()
    
          
    
    @tf.function
    def compute_cost(self, AL, Y):
        outer_activation = self.layers[len(self.layers) - 1].activation
        if outer_activation == "sigmoid":
          loss = tf.keras.losses.BinaryCrossentropy()
        elif outer_activation == "softmax":
          loss = tf.keras.losses.CategoricalCrossentropy()
        return loss(Y, AL)
    
    @tf.function
    def forward(self, X):
        # X- batch of dim (n, m)
        # n -number of layer in input layer
        # m- number of samples in batch
        out_prev = X
        for layer in self.layers:
            out_prev = layer.forward(out_prev,self.epsilon)

        return out_prev
    
    @tf.function
    def backward(self, X, Y):
        # revered layers
        # Al - Model label (n, m)
        # Y -- true "label" (n, m)
        # t -- current epoch
        dout_prev = self.layers[-1].backward_out(Y)
        for l in reversed(range(len(self.layers[:-1]))):
            layer = self.layers[l]
            dout_prev = layer.backward(dout=dout_prev)

    @tf.function
    def update_params(self, t):
        for layer in self.layers:
            if hasattr(layer, 'init_params'):
              layer.update(self.learning_rate,self.epsilon,t)

    def predict(self,X, Y=None):
        #     This function is used to predict the results of a  n-layer neural network.

        # Arguments:
        # X -- data set of examples you would like to label
        # parameters -- parameters of the trained model

        # Returns:
        # p -- predictions for the given dataset X
        out_prev = tf.transpose(tf.cast(tf.reshape(X, shape=[X.shape[0], -1]), tf.float32))
        
        m = X.shape[1]
        for layer in self.layers:
            out_prev = layer.forward(out_prev, self.epsilon, False)

        p = tf.zeros_like(out_prev)

        activation = self.layers[len(self.layers) - 1].activation

        if activation == "sigmoid":
            p = tf.where(out_prev > 0.5, tf.ones_like(p), tf.zeros_like(p))
        elif activation == "softmax":
          pred_max = tf.argmax(out_prev, axis=0) 
          indices = tf.stack([pred_max, tf.range(m)], axis=1)  # Shape: [batch_size, 2]
          p = tf.tensor_scatter_nd_update(p, indices, tf.ones([m])) 
        else:
            pass

        if not Y is None:
            Y = tf.transpose(tf.cast(Y, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(tf.equal(p, Y), tf.float32)).numpy()
            print(f"Accuracy: {accuracy}")

        return tf.cast(p ,dtype=tf.int32)
