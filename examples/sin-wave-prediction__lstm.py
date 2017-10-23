import sys
sys.path.append("../src/")
import LSTM_module2 as lstm
import modules as mod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import ipdb
def mean_squared_error(x, y, name):
    return tf.square(x - y, name=name)


CELL_SIZE = 100
go_back = 5
inp = mod.ConstantPlaceholderModule("input", shape=(1, 1))
label = mod.ConstantPlaceholderModule("label", shape=(1, 1))
cell = lstm.LSTM_Cell("lstm_cell", 1, CELL_SIZE)
lin_class = mod.FullyConnectedLayerModule("linear_classifier", tf.identity, CELL_SIZE, 1)
err = mod.ErrorModule("mse", mean_squared_error)
opt = mod.OptimizerModule("gradient_descent", tf.train.GradientDescentOptimizer(0.1))


cell.add_input(inp)
lin_class.add_input(cell)
err.add_input(label)
err.add_input(lin_class)
opt.add_input(err)

opt.create_output(go_back)


N = 2000
M = 10


def f(t):
    return np.sin(t)


with tf.Session() as sess:
    error = []
    sess.run(tf.global_variables_initializer())
    for t in np.linspace(0, 2 * np.pi * M, M * N):
        res = sess.run(opt.outputs[go_back], feed_dict={inp.placeholder: [[t]], label.placeholder: [[f(t)]]})
        error.append(res[0][0])
        print(np.mean(np.abs(res)), t / (2 * np.pi), end="\r")
    print(res[0][0], "shape", res.shape)
    plt.figure()
    plt.subplot(211)
    plt.plot(np.array(error), 'b-')
    plt.subplot(212)
    plt.plot(np.linspace(0, 2 * np.pi * M, M * N), f(np.linspace(0, 2 * np.pi * M, M * N)))
    plt.show()

