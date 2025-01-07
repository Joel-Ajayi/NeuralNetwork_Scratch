import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

@tf.function
def normalize_data(z, epsilon):
    mu = tf.reduce_mean(z, axis=1, keepdims=True)  # Mean across batch (axis 1)
    var = tf.reduce_mean(tf.pow((z - mu), 2), axis=1, keepdims=True)  # Variance across batch (axis 1)
    var_eps_sum = var + epsilon
    z_mu_diff = z - mu
    z_norm = z_mu_diff * tf.math.rsqrt(var_eps_sum)
    return (
        mu,
        var,
        z_norm,
        z_mu_diff,
        var_eps_sum,
    )


@tf.function
def normalize_data_with_var(z, mu, var, epsilon=10e-10):
    z_norm = tf.divide(tf.subtract(z,mu) , tf.sqrt(tf.add(var,epsilon)))
    return z_norm


def plot_decision_boundary(model, X, Y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.vstack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.scatter(X[0, :], X[1, :], c=Y, edgecolors="k", cmap=plt.cm.coolwarm)
    plt.title("Multi-Class and Binary Decision Boundaries in PCA Space")
    plt.ylabel("Principal Component 2")
    plt.xlabel("Principal Component 1")
    plt.show()
