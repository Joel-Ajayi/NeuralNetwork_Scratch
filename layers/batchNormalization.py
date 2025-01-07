from numpy import ptp
import tensorflow as tf
from utils import normalize_data, normalize_data_with_var

class BatchNorm():
    def __init__(self, momentum=0.9):
      self.momentum = tf.constant(momentum)
      self.units = 0

    @property
    def variables(self):
      return [self.gamma, self.beta]

    def init_batch_params(self, m, prev_units):
      self.z_norm = tf.Variable(tf.zeros([self.units, m]), trainable=False, dtype=tf.float32)
      self.z_mu_diff = tf.Variable(tf.zeros([self.units, m]), trainable=False, dtype=tf.float32)
      self.var_eps_sum = tf.Variable(tf.zeros([self.units, 1]), trainable=False, dtype=tf.float32)
    
    def init_params(self,prev_units, optimizer):
      self.units = prev_units
      self.opt = optimizer

      self.beta = tf.Variable(tf.zeros([self.units, 1]),trainable=True)
      self.gamma = tf.Variable(tf.ones([self.units, 1]),trainable=True)

      self.dgamma = tf.Variable(tf.zeros_like(self.gamma),trainable=False)
      self.dbeta = tf.Variable(tf.zeros_like(self.beta),trainable=False)  

      self.mu_running = tf.Variable(tf.zeros([self.units, 1]),trainable=False)
      self.var_running = tf.Variable(tf.zeros([self.units, 1]),trainable=False)

      if optimizer == "momentum":
        self.beta1 = tf.constant(0.9)
        self.dVgamma = tf.Variable(tf.zeros_like(self.gamma),trainable=False)
        self.dVbeta = tf.Variable(tf.zeros_like(self.beta),trainable=False)      
      
      elif optimizer == "adam":
        self.beta1 = tf.constant(0.9)
        self.beta2 = tf.constant(0.999)

        self.dVgamma = tf.Variable(tf.zeros_like(self.gamma), trainable=False)
        self.dVgamma_c = tf.Variable(tf.zeros_like(self.gamma), trainable=False)

        self.dVbeta = tf.Variable(tf.zeros_like(self.beta), trainable=False)
        self.dVbeta_c = tf.Variable(tf.zeros_like(self.beta), trainable=False)

        self.dSgamma = tf.Variable(tf.zeros_like(self.gamma), trainable=False)
        self.dSgamma_c = tf.Variable(tf.zeros_like(self.gamma), trainable=False)

        self.dSbeta = tf.Variable(tf.zeros_like(self.beta), trainable=False)
        self.dSbeta_c = tf.Variable(tf.zeros_like(self.beta), trainable=False)
    
    @tf.function
    def forward(self, out_prev, eps, is_training=True):
      # normalize z
      if is_training:
        mu, var, z_norm, z_mu_diff, var_eps_sum = normalize_data(out_prev, eps)
         # update running mean
        self.z_norm.assign(z_norm)
        self.z_mu_diff.assign(z_mu_diff)
        self.var_eps_sum.assign(var_eps_sum)
        self.mu_running.assign(self.momentum * self.mu_running + (1 - self.momentum) * mu)
        self.var_running.assign(self.momentum * self.var_running + (1 - self.momentum) * var)
      else:
        z_norm = normalize_data_with_var(tf.identity(out_prev), self.mu_running, self.var_running, eps)

      # scale and shift z_norm with gamma and beta
      out = tf.add(tf.multiply(self.gamma,z_norm) ,self.beta)
      return out
    
    @tf.function
    def backward(self, dout):
      m = tf.cast(dout.shape[1], tf.float32)

      self.dgamma.assign(tf.reduce_sum(self.z_norm * dout, axis=1, keepdims=True))
      self.dbeta.assign(tf.reduce_sum(dout, axis=1, keepdims=True))
      dz_norm = dout * self.gamma

      ivar = tf.math.rsqrt(self.var_eps_sum)
      dout_prev = tf.divide(tf.multiply(self.gamma,ivar), m) * (
                  tf.multiply(m ,dz_norm)
                - tf.reduce_sum(dz_norm, axis=1, keepdims=True)
                - (tf.pow(ivar,2) * self.z_mu_diff * tf.reduce_sum(dz_norm * self.z_mu_diff, axis=1, keepdims=True))
            )

      return dout_prev

    @tf.function
    def update(self, learning_rate, eps, t):
        # apply optimisers
        self.optimize(t)

        if self.opt == "gd":
            # Update gamma in hidden layers with BN
            self.gamma.assign_sub(learning_rate * self.dgamma)
            # Update beta in hidden layers with BN
            self.beta.assign_sub(learning_rate * self.dbeta)
        elif self.opt == "momentum":
            # Update gamma in hidden layers with BN
            self.gamma.assign_sub(learning_rate * self.dVgamma)
            # Update beta in hidden layers with BN

            self.beta.assign_sub(learning_rate * self.dVbeta)
        elif self.opt == "adam":
            self.beta.assign_sub(learning_rate * tf.divide(self.dVbeta_c, tf.add(eps, tf.sqrt(self.dSbeta_c))))
            self.gamma.assign_sub(learning_rate * tf.divide(self.dVgamma_c, tf.add(eps, tf.sqrt(self.dSgamma_c))))
      
    @tf.function
    def optimize(self, t):
      if self.opt == "momentum":
          self.optimize_momentum()
      elif self.opt == "adam":
          self.optimize_adam(t)
    
    @tf.function
    def optimize_momentum(self):
      self.dVgamma.assign((self.beta1 * self.dVgamma) + ((1 - self.beta1) * self.dgamma))
      self.dVbeta.assign((self.beta1 * self.dVbeta) + ((1 - self.beta1) * self.dbeta))
    
    @tf.function
    def optimize_adam(self, t):
      beta1 = self.beta1
      beta2 = self.beta2

      # Moving average of the gradients and corrections
      self.dVgamma.assign((beta1 * self.dVgamma) + ((1 - beta1) * self.dgamma))
      self.dVgamma_c.assign(self.dVgamma / (1 - beta1**t))

      self.dVbeta.assign((beta1 * self.dVbeta) + ((1 - beta1) * self.dbeta))
      self.dVbeta_c.assign(self.dVbeta / (1 - beta1**t))

        # Moving average of the squared gradients. for weights, beta and gamma
      self.dSgamma.assign((beta2 * self.dSgamma) + ((1 - beta2) * (self.dgamma**2)))
      self.dSgamma_c.assign(self.dSgamma / (1 - beta2**t))

      self.dSbeta.assign((beta2 * self.dSbeta) + ((1 - beta2) * (self.dbeta**2)))
      self.dSbeta_c.assign(self.dSbeta / (1 - beta2**t))


            

    



