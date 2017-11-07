import sys
sys.path.append("../src/")
import LSTM_module2 as lstm
import modules as mod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from lstm_plotting import Plot
import lstm_functions as func

import ipdb


def mean_squared_error(x, y, name):
    return tf.square(x - y, name=name)

def test_epoch(sess, epoch, save_name="spectrogram_test"):
    error_total = np.array([])
    prediction_total = np.empty([0, out_size])
    save_name = save_name+"lstm_test_{}".format(epoch+1)
    feed_dict = {}
    feed_dict[inp.placeholder] = testX
    feed_dict[target.placeholder] = testY
    for t in range(testX.shape[0]//BATCH_SIZE - 1):    # 2000
        error, prediction = sess.run([opt.outputs[TIME_DEPTH], out_prediction.outputs], feed_dict={inp.placeholder: testX[t*BATCH_SIZE : (t+1)*BATCH_SIZE], target.placeholder: testY[t*BATCH_SIZE : (t+1)*BATCH_SIZE]})
        error_total = np.append(error_total, error.ravel())
        prediction_total = np.vstack((prediction_total, prediction[1]))
        
        if t%10 == 0:
            sys.stdout.write("\r" + "epoch process{}%".format((t*BATCH_SIZE / testX.shape[0])*100))
            sys.stdout.flush()

    myplot.plot_compare_specgram(seg_time_test[delay:prediction_total.shape[0]+1],
                freqs_test, prediction_total.T, seg_time_test[delay:], freqs_test, testY.T,
                save_name=save_name+"compare spectrogram_test.png", ifsave=True)
    
WINDOW = 1
    
audio_names_train = ['sp11.wav','sp12.wav']   # '04_Chapter_1-4.wav'
audio_names_test = ['sp10.wav']   #,'05_Chapter_1-5.wav'
s_train, seg_time_train, freqs_train = func.concat_spectrogram(audio_names_train, save_name="train11+12-on-M10-train_concat_spec")
s_test, seg_time_test, freqs_test = func.concat_spectrogram(audio_names_test, save_name="train11+12-on-M10-test_concat_spec")

reformdata_train = func.rolling(s_train, window=WINDOW)
reformdata_test = func.rolling(s_test, window=WINDOW)

delay = 1
        
trainX = reformdata_train[:-WINDOW-delay]
trainY = reformdata_train[WINDOW+delay:, 0:s_train.shape[1]]
testX = reformdata_test[:-WINDOW-delay]
testY = reformdata_test[WINDOW+delay:, 0:s_test.shape[1]]

CELL_SIZE = 300
TIME_DEPTH = 5
BATCH_SIZE = 1
NFFT = 128
in_size = (NFFT + 1) * WINDOW
out_size = NFFT + 1

inp = mod.ConstantPlaceholderModule("input", shape=(BATCH_SIZE, in_size))
target = mod.ConstantPlaceholderModule("target", shape=(BATCH_SIZE, out_size))
cell = lstm.LSTM_Cell("lstm_cell", in_size, CELL_SIZE)

out_prediction = mod.FullyConnectedLayerModule("out_prediction", tf.identity, CELL_SIZE, out_size)
err = mod.ErrorModule("mse", mean_squared_error)
opt = mod.OptimizerModule("adam", tf.train.AdamOptimizer())

#  Connect input
cell.add_input(inp)
out_prediction.add_input(cell)
err.add_input(target)
err.add_input(out_prediction)
opt.add_input(err)
opt.create_output(TIME_DEPTH)
out_prediction.create_output(1)

myplot = Plot()

train_length = trainX.shape[0] #2000#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        error_total = np.array([])
        prediction_total = np.empty([0, out_size])
        save_name = "lstm_{}_epoch_sp11+12-ON-sp10_".format(epoch+1)
        error = 0
        for t in range(train_length//BATCH_SIZE - 1):
            error += sess.run(opt.outputs[TIME_DEPTH], feed_dict={inp.placeholder: trainX[t*BATCH_SIZE : (t+1)*BATCH_SIZE], target.placeholder: trainY[t*BATCH_SIZE : (t+1)*BATCH_SIZE]})
            prediction = sess.run(out_prediction.outputs, feed_dict={inp.placeholder: trainX[t*BATCH_SIZE : (t+1)*BATCH_SIZE], target.placeholder: trainY[t*BATCH_SIZE : (t+1)*BATCH_SIZE]})
            prediction_total = np.vstack((prediction_total, prediction[1]))
            if t == train_length//BATCH_SIZE - 2:
                sys.stdout.write("\r" + "epoch process{}%".format((t*BATCH_SIZE / train_length)*100))
                sys.stdout.flush()
        
        error_total = np.append(error_total, np.sum(error))
        # plot the compare of two spectrograms
        ipdb.set_trace()
        myplot.plot_compare_specgram(seg_time_train[delay:prediction_total.shape[0]+1], freqs_train, prediction_total.T, seg_time_train[delay:trainY.shape[0]+1], freqs_train, trainY.T, save_name=save_name+"compare spectrogram_train.png", ifsave=True)

        test_epoch(sess, epoch, save_name=save_name)
        print ("Error:{}".format(error_total))

ipdb.set_trace()
plt.figure()
myplot.plot_score(np.array(error_total), title="Score on training steps", x="steps", y="MSE", save_name=save_name+"MSE in training steps.png")




    
