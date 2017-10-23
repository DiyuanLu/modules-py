import sys
sys.path.append("../src/")
import LSTM_module2 as lstm
import modules as mod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
#import ipdb
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

audio_names_train = ['sp10.wav', 'sp11.wav', 'sp12.wav', 'sp13.wav']
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
label = mod.ConstantPlaceholderModule("label", shape=(1, 1))
cell = lstm.LSTM_Cell("lstm_cell", 1, CELL_SIZE)
lin_class = mod.FullyConnectedLayerModule("linear_classifier", tf.identity, CELL_SIZE, 1)
err = mod.ErrorModule("mse", mean_squared_error)
opt = mod.OptimizerModule("adam", tf.train.AdamOptimizer())

#  Connect input
cell.add_input(inp)
lin_class.add_input(cell)
err.add_input(label)
err.add_input(lin_class)
opt.add_input(err)
opt.create_output(TIME_DEPTH)


#def test_epoch(sess):
    #acc = 0
    #for j in range(5000//BATCH_SIZE - 1):
        #batch = test_mnist[j*BATCH_SIZE : (j+1)*BATCH_SIZE]
        #batch_labels = test_mnist_label[j*BATCH_SIZE : (j+1) * BATCH_SIZE]
        #feed_dict = {}
        #feed_dict[inp.placeholder] = batch
        #feed_dict[labels.placeholder] = to_one_hot(batch_labels)
        #acc += sess.run(accuracy.outputs[TIME_DEPTH], feed_dict=feed_dict)
        #print("accuracy:\t{:.2f} %\r".format(100 * acc / (j+1)))
    #print("")


N_EPOCH = 1000

with tf.Session() as sess:
    error = []
    sess.run(tf.global_variables_initializer())
    for t in range(N_EPOCH):
        res = sess.run(opt.outputs[TIME_DEPTH], feed_dict={inp.placeholder: [[s_train[t]]], label.placeholder: [[s_test[t]]]})
        error.append(res[0][0])
        if t%2 == 0:
            sys.stdout.write("\r" + "epoch process{}%".format((t / N_EPOCH)*100))
            sys.stdout.flush()
            #test_epoch(sess)
        
    plt.figure()
    plt.plot(np.array(error), 'c-')
    plt.title("MSE with LSTM cell")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.savefig("audio_lstm_error.png", format="png")
    plt.show()
    
