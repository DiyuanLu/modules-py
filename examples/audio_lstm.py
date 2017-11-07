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

import ipdb
import lstm_functions as func


def mean_squared_error(x, y, name):
    return tf.square(x - y, name=name) 

def log_loss(x, y, name):
    return tf.losses.log_loss(y, x)

def test_epoch(sess, epoch, sampFreq_test, save_name):
    error_total = np.array([])
    prediction_total = np.array([])
    save_name = save_name+"lstm_test_{}".format(epoch+1)
    feed_dict = {}
    feed_dict[inp.placeholder] = testX
    feed_dict[target.placeholder] = testY
    for t in range(testX.shape[0]//BATCH_SIZE - 1):    # 2000
        error, prediction = sess.run([opt.outputs[TIME_DEPTH], out_prediction.outputs], feed_dict={inp.placeholder: testX[t*BATCH_SIZE : (t+1)*BATCH_SIZE], target.placeholder: testY[t*BATCH_SIZE : (t+1)*BATCH_SIZE]})
        error_total = np.append(error_total, error.ravel())
        prediction_total = np.append(prediction_total, prediction[1].ravel())
        
        if t%10 == 0:
            sys.stdout.write("\r" + "epoch process{}%".format((t*BATCH_SIZE / testX.shape[0])*100))
            sys.stdout.flush()
                
    plt.figure()
    myplot.plot_compare(testY[0:prediction_total.shape[0]], prediction_total, title="Prediction in test", x="time", y="amplitude", save_name=save_name+"_test.png" )
    playback = func.data2int16(np.array(prediction_total))
    func.play_wav(playback, sampFreq_test)
    func.save_wav(playback, sampFreq_test, save_name+"_test.wav")
    

audio_names_train = ['sp10.wav','sp11.wav', 'sp12.wav']   #'04_Chapter_1-4.wav'  , ,
audio_names_test = ['sp13.wav']       #'05_Chapter_1-5.wav'
s_train, sampFreq0 = func.concat_audios(audio_names_train, ifnorm=True, save_name="10~12-ON-13-train_concat", ifsave=False)
s_test, sampFreq_test = func.concat_audios(audio_names_test, ifnorm=True, save_name="10~12-ON-13-test_concat", ifsave=False)

trainX = s_train[0:-1].reshape(-1, 1)
trainY = s_train[1:  ].reshape(-1, 1)
testX = s_test[0:-1].reshape(-1, 1)
testY = s_test[1: ].reshape(-1, 1)

CELL_SIZE = 500
TIME_DEPTH = 5
BATCH_SIZE = 200

inp = mod.ConstantPlaceholderModule("input", shape=(BATCH_SIZE, 1))
target = mod.ConstantPlaceholderModule("target", shape=(BATCH_SIZE, 1))
cell = lstm.LSTM_Cell("lstm_cell", 1, CELL_SIZE)
out_prediction = mod.FullyConnectedLayerModule("out_prediction", tf.identity, CELL_SIZE, 1)
err = mod.ErrorModule("mse", mean_squared_error)
#err = mod.ErrorModule("logloss", log_loss)
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
    for epoch in range(20):
        error_total = np.array([])
        prediction_total = np.array([])
        save_name = "lstm_{}_epoch_batchsize_{}_10~12-ON-13_".format(epoch+1, BATCH_SIZE)
        error = 0
        for t in range(train_length//BATCH_SIZE - 1):

            error += sess.run(opt.outputs[TIME_DEPTH], feed_dict={inp.placeholder: trainX[t*BATCH_SIZE : (t+1)*BATCH_SIZE], target.placeholder: trainY[t*BATCH_SIZE : (t+1)*BATCH_SIZE]})
            prediction = sess.run(out_prediction.outputs, feed_dict={inp.placeholder: trainX[t*BATCH_SIZE : (t+1)*BATCH_SIZE], target.placeholder: trainY[t*BATCH_SIZE : (t+1)*BATCH_SIZE]})
            prediction_total = np.append(prediction_total, prediction[1].ravel())
            if t == train_length//BATCH_SIZE - 2:
                sys.stdout.write("\r" + "epoch process{}%".format((t*BATCH_SIZE / train_length)*100))
                sys.stdout.flush()
        
        error_total = np.append(error_total, np.sum(error))
        plt.figure()
        myplot.plot_compare(np.ravel(trainY[0:prediction_total.shape[0]]), prediction_total, title="Prediction in train", x="time", y="amplitude", save_name=save_name+"_train.png" )
        plt.close()

        playback = func.data2int16(np.array(prediction_total))
        func.play_wav(playback, sampFreq0)
        func.save_wav(playback, sampFreq0, save_name+"_train.wav")
        test_epoch(sess, epoch, sampFreq_test, save_name)
        print ("Error:{}".format(error_total))
   
plt.figure()
myplot.plot_score(np.array(error_total), title="Score on training steps", x="epochs", y="MSE", save_name=save_name+"MSE in training steps.png", ifsave=True)




    
