import sys
sys.path.append("../src/")
import LSTM_module2 as lstm
import modules as mod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from plotting import Plot

import ipdb
def mean_squared_error(x, y, name):
    return tf.square(x - y, name=name)


def playback_transform(data, sampFreq):
    """transform network output to playable audio signal int16
    play it and save as wav file"""
    if data.dtype != 'int16':
        playback = np.int16(np.round(data*(2**15)))

    return playback

def concat_audios(audios, ifnorm=True, save_name="train_concat"):
    '''
    Param:
    a list of audio names in the folder original_audio'''
    snd_total = []
    for arg in audios:
        sampFreq, snd = wavfile.read('original_audio/'+arg)
        snd_total = np.append(snd_total, snd)
    #wavfile.write(save_name+".wav", sampFreq, np.int16(snd_total))
    # Normalize the amplitude data
    if ifnorm:
        snd_total = snd_total / (2.**15)
    return np.array(snd_total), sampFreq

def test_epoch(sess):
    error_total = []
    prediction_total = []
    save_name = "lstm_test"
    feed_dict = {}
    feed_dict[inp.placeholder] = testX
    feed_dict[target.placeholder] = testY
    for t in range(testX.shape[0]):
        error, prediction = sess.run([opt.outputs[TIME_DEPTH], out_prediction.outputs], feed_dict=feed_dict)
        error_total.append(error[0][0])
        prediction_total.append(prediction[1][0][0])
        if t%10 == 0:
            sys.stdout.write("\r" + "epoch process{}%".format((t / testX.shape[0])*100))
            sys.stdout.flush()
                
    plt.figure()
    myplot.plot_compare(testY, np.array(prediction_total), title="Prediction in test", x="time", y="amplitude", save_name=save_name+".png" )
    playback = playback_transform(np.array(prediction_total), sampFreq0)
    sd.play(playback, sampFreq0)
    wavfile.write(save_name+".wav", sampFreq0, playback)
    
    print("")

audio_names_train = ['sp10.wav']   #'sp12.wav', 'sp13.wav', 'sp11.wav', 
audio_names_test = ['sp14.wav']
s_train, sampFreq0 = concat_audios(audio_names_train, ifnorm=True, save_name="4-on-1-train_concat")
s_test, sampFreq1 = concat_audios(audio_names_test, ifnorm=True, save_name="4-on-1-test_concat")
    
trainX = s_train[0:-1].reshape(-1, 1)
trainY = s_train[1:  ].reshape(-1, 1)
testX = s_test[0:-1].reshape(-1, 1)
testY = s_test[1: ].reshape(-1, 1)

CELL_SIZE = 100
TIME_DEPTH = 5
inp = mod.ConstantPlaceholderModule("input", shape=(1, 1), dtype="float32")
target = mod.ConstantPlaceholderModule("target", shape=(1, 1))
cell = lstm.LSTM_Cell("lstm_cell", 1, CELL_SIZE)
out_prediction = mod.FullyConnectedLayerModule("out_prediction", tf.identity, CELL_SIZE, 1)
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

length = trainX.shape[0]
BATCH_SIZE = 200
with tf.Session() as sess:
    error_total = []
    prediction_total = []
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
        save_name = "lstm_{}_epoch_train_predict".format(epoch+1)
        for t in range(length):
            #ipdb.set_trace()
            error, prediction = sess.run([opt.outputs[TIME_DEPTH], out_prediction.outputs], feed_dict={inp.placeholder: [trainX[t]], target.placeholder: [trainY[t]]})
            error_total.append(error[0][0])
            prediction_total.append(prediction[1][0][0])
            if t%2 == 0:
                sys.stdout.write("\r" + "epoch process{}%".format((t / length)*100))
                sys.stdout.flush()
                #
    ipdb.set_trace()
    plt.figure()
    myplot.plot_compare(trainY, np.array(prediction_total), title="Prediction in test", x="time", y="amplitude", save_name=save_name+".png" )
    playback = playback_transform(np.array(prediction_total), sampFreq0)
    sd.play(playback, sampFreq0)
    wavfile.write(save_name+".wav", sampFreq0, playback)
    
    test_epoch(sess)
    plt.show()



plt.figure()
myplot.plot_score(np.array(error_total), length, title="Score on training steps", x="steps", y="MSE", save_name=save_name+"MSE in training steps.png")

plt.show()


    
