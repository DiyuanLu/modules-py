import sys
sys.path.append("../src/")
import modules as mod
import LSTM_module2 as lstm
import get_mnist as gm
import tensorflow as tf
import numpy as np


class LSTM_Cell(mod.ComposedModule):
    '''Define LSTM net'''
    def define_inner_modules(self, name, in_size, cell_size):
        """Typical LSTM cell with three gates. Detailed tutorial see
        http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        """
        self.in_size = in_size
        self.cell_size = cell_size
        self.input_module = mod.ConcatModule("concat", 1, in_size + cell_size)
        # Three gates for input, output, cell state
        f_t = mod.FullyConnectedLayerModule("f_t", tf.sigmoid, in_size + cell_size, cell_size)
        i_t = mod.FullyConnectedLayerModule("i_t", tf.sigmoid, in_size + cell_size, cell_size)
        o_t = mod.FullyConnectedLayerModule("o_t", tf.sigmoid, in_size + cell_size, cell_size)

        # transformed input and last time hidden-state
        CHat_t = mod.FullyConnectedLayerModule("CHat_t", tf.tanh, in_size + cell_size, cell_size)
        # cell states related
        self.C_t = mod.AddModule("C_t")
        tanh_C_t = mod.ActivationModule("tanh_C_t", tf.tanh)
        # residual of last time cell state
        state_residual = mod.EleMultiModule("state_residual")
        state_update = mod.EleMultiModule("state_update")   # i_t(eleMulti)CHat_t

        # hidden states
        self.h_t = mod.EleMultiModule("h_t")

        # making connections
        self.input_module.add_input(self.h_t, -1)
        f_t.add_input(self.input_module)
        state_residual.add_input(self.C_t, -1)
        state_residual.add_input(f_t, 0)

        i_t.add_input(self.input_module)
        CHat_t.add_input(self.input_module)
        state_update.add_input(i_t, 0)
        state_update.add_input(CHat_t, 0)

        self.C_t.add_input(state_update)  # C_t = state_redidual + state_update
        self.C_t.add_input(state_residual)

        o_t.add_input(self.input_module)
        tanh_C_t.add_input(self.C_t)
        self.h_t.add_input(o_t, 0)
        self.h_t.add_input(tanh_C_t, 0)

        # set input and output
        self.output_module = self.h_t


def cross_entropy(a, b, name):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=b, name=name))

def to_one_hot(num_classes, labels):
    '''Make int label into one-hot encoding
    Param:
    num_classes: int, number of classes
    labels: 1D array''' 
    ret = np.eye(num_classes)[labels]

    return ret

BATCH_SIZE = 500
TIME_DEPTH = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10

train_images_name = "train-images-idx3-ubyte.gz"  #  training set images (9912422 bytes)
train_data_filename = gm.maybe_download(train_images_name)
train_mnist = gm.extract_data(train_data_filename, 60000)

train_label_name = "train-labels-idx1-ubyte.gz"  #  training set labels (28881 bytes)
train_label_filename = gm.maybe_download(train_label_name)
train_mnist_label = gm.extract_labels(train_label_filename, 60000)

test_image_name = "t10k-images-idx3-ubyte.gz"  #  test set images (1648877 bytes)
test_data_filename = gm.maybe_download(test_image_name)
test_mnist = gm.extract_data(test_data_filename, 5000)

test_label_name = "t10k-labels-idx1-ubyte.gz"  #  test set labels (4542 bytes)
test_label_filename = gm.maybe_download(test_label_name)
test_mnist_label = gm.extract_labels(test_label_filename, 5000)


inp = mod.ConstantPlaceholderModule("input", shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1))
labels = mod.ConstantPlaceholderModule("input_labels", shape=(BATCH_SIZE, NUM_CLASSES))

cell_size = 200
network = LSTM_Cell("lstm", IMG_HEIGHT, cell_size)
#
one_time_error = mod.ErrorModule("cross_entropy", cross_entropy)
error = mod.TimeAddModule("add_error")
accuracy = mod.BatchAccuracyModule("accuracy")
optimizer = mod.OptimizerModule("adam", tf.train.AdamOptimizer())
#
network.add_input(inp)
one_time_error.add_input(network)
one_time_error.add_input(labels)
error.add_input(one_time_error, 0)
error.add_input(error, -1)
accuracy.add_input(network)
accuracy.add_input(labels)
optimizer.add_input(error)
optimizer.create_output(TIME_DEPTH)
accuracy.create_output(TIME_DEPTH)
#


def train_batch(sess, i):
    batch = train_mnist[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    batch_labels = train_mnist_label[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    feed_dict = {}
    feed_dict[inp.placeholder] = batch
    feed_dict[labels.placeholder] = to_one_hot(batch_labels)
    err = sess.run(optimizer.outputs[TIME_DEPTH], feed_dict=feed_dict)
    print("error:\t\t{:.4f}".format(err), end='\r')


def test_epoch(sess):
    print("")
    acc = 0
    for j in range(5000//BATCH_SIZE - 1):
        batch = test_mnist[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
        batch_labels = test_mnist_label[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
        feed_dict = {}
        feed_dict[inp.placeholder] = batch
        feed_dict[labels.placeholder] = to_one_hot(batch_labels)
        acc += sess.run(accuracy.outputs[TIME_DEPTH], feed_dict=feed_dict)
        print("accuracy:\t{:.2f} %".format(100 * acc / (j+1)), end='\r')
    print("")


#N_EPOCH = 2
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    for epoch_number in range(N_EPOCH):
#        for i in range(60000//BATCH_SIZE - 1):
#            if i % 100 == 0:
#                test_epoch(sess)
#            train_batch(sess, i)
