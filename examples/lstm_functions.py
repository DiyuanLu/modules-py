import scipy.signal as signal
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#from librosa.feature import mfcc
from python_speech_features import mfcc
import ipdb

def data2int16(data):
    """transform network output to playable audio signal int16
    play it and save as wav file"""
    if data.dtype != 'int16':
        playback = np.int16(np.round(data*(2**15)))

    return playback


def concat_audios(audios, ifnorm=True, save_name="train_concat", ifsave=False):
    '''
    Param:
    a list of audio names in the folder original_audio'''
    snd_total = []
    for arg in audios:
        sampFreq, snd = wavfile.read('original_audio/'+arg)
        if sampFreq == 8000:
            snd_total = np.append(snd_total, snd)
        else:
            snd_resample = resample(arg, sampFreq, 8000, save_name=arg+"reample2-8000")
            snd_total = np.append(snd_total, snd_resample)
    if ifsave:
        wavfile.write(save_name+".wav", sampFreq, np.int16(snd_total))
    # Normalize the amplitude data
    if ifnorm:
        snd_total = snd_total / (2.**15)
    return np.array(snd_total), sampFreq

def resample(audio, in_rate, out_rate, save_name="train_concat"):
    '''
    Parameter;
    audio: array_like--The data to be resampled.
    in_rate: the sampling rate of original data
    out_rate: the target sampling rate you want to convert to
    '''
    in_rate, snd = wavfile.read('original_audio/'+audio)
    num = snd.size / in_rate * out_rate
    resample_snd = signal.resample(snd, num)
    wavfile.write('original_audio/'+save_name+".wav", out_rate, np.int16(resample_snd))
    return resample_snd

def save_wav(data, sampFreq, save_name="saved wav file"):
    '''
    Param;
    data: NumPy array
    sampFreq: sampling frequency'''
    wavfile.write(save_name+".wav", sampFreq, data)

def play_wav(data, sampFreq):
    '''
    Param:
    data: NumPy array
    sampFreq: sampling frequency
    '''
    sd.play(data, sampFreq)

def get_autocorr_in_dir(file_dir): #
    """
    read label files in label_dir and same them in a dictionary
    Parameter:
        file_dir: path of directory with audio files
    Return:
        label_dic: the dic of all the label files: {"file_name" : label details}
    """
    file_names = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f)) and '.wav' in f]
    
    for file_name in file_names:
        print (file_name)
        sampFreq, snd = wavfile.read('original_audio/non-speech/'+file_name)
        snd = snd / (2.**15)
        err = snd - np.mean(snd)
        norm = np.sum(err**2)
        temp = np.correlate(err, err, mode='full')/norm
        result = temp[temp.size/2:]

        plt.figure()
        plt.subplot(211)
        plt.plot(result, 'b*-')
        plt.title(os.path.splitext(os.path.basename(file_name))[0]+" zoomed out")
        plt.xlabel("delays")
        plt.ylabel("correlation")
        plt.xlim([-2, len(result)])
        plt.subplot(212)
        plt.plot(result[0:2000], 'm*-')
        plt.title(os.path.splitext(os.path.basename(file_name))[0]+" zoomed in")
        plt.xlabel("delays")
        plt.ylabel("correlation")
        plt.xlim([-2, 2000])
        plt.tight_layout()
        #ipdb.set_trace()
        plt.savefig(os.path.splitext(os.path.basename(file_name))[0]+"autocorrelation.png", format="png")
        plt.close()

def resample_in_dir(file_dir): #
    """
    read label files in label_dir and same them in a dictionary
    Parameter:
        file_dir: path of directory with audio files
    Return:
        label_dic: the dic of all the label files: {"file_name" : label details}
    """
    file_names = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f)) and '.wav' in f]
    target_rate = 8000
    for file_name in file_names:
        print (file_name)
        ipdb.set_trace()
        save_name = os.path.splitext(os.path.basename(file_name))[0]
        # read file
        sampFreq, snd = wavfile.read(file_dir+file_name)
        if sampFreq != target_rate:
            # downsample the audio
            if len(snd.shape) != 1:   # check whether the data is 2D or 3D
                data = resample(snd[:,0], sampFreq, target_rate, save_name=file_dir+save_name+"_downsampling.wav")
         
        
def concat_MFCC(audio_names, save_name="train_concat", ifsave=False):
    '''
    Param:
        audio_names: a list of audio names in the folder original_audio
    Return:
        MFCC: 2D array, frames*NUMCEP
    '''
    audio_dir = 'original_audio/'
    NUMCEP = 25
    MFCC = np.empty((NUMCEP, 0))
    for arg in audio_names:
        sampFreq, snd = wavfile.read(audio_dir+arg)
        if sampFreq == 8000:
            current_MFCC = mfcc(y=snd, sr=sampFreq, n_mfcc=NUMCEP)
            MFCC = np.hstack((MFCC, current_MFCC))
        else:
            snd_resample = resample(arg, sampFreq, 8000, save_name=arg+"reample2-8000")
            current_MFCC = mfcc(y=snd_resample, sr=8000, n_mfcc=NUMCEP)
            MFCC = np.hstack((MFCC, current_MFCC))
        #if ifsave:
            #pickle it
    # Normalize the amplitude data
    return MFCC.T

def concat_spectrogram(audio_names, save_name="concat spectrogram"):
    '''
    Param:
        audio_names: a list of audio names in the folder original_audio
    Return:
        spectrogram: 2D array: segment times*(NFFT+1)
    '''
    audio_dir = 'original_audio/'
    NFFT = 128
    spectrogram = np.empty((NFFT+1, 0))
    seg_time = np.empty((1, 0))
    for arg in audio_names:
        sampFreq, snd = wavfile.read(audio_dir+arg)
        if sampFreq == 8000:
            freq, t, Sxx = signal.spectrogram(snd, fs=8000)
            spectrogram = np.hstack((spectrogram, Sxx))
            seg_time = np.append(seg_time, t)
        else:
            snd_resample = resample(arg, sampFreq, 8000, save_name=arg+"reample2-8000")
            freq, t, Sxx = signal.spectrogram(snd_resample, fs=8000)
            spectrogram = np.hstack((spectrogram, Sxx))
            seg_time = np.append(seg_time, t)
        # nice spectrogram is scaled to 10*log()
        
#    plt.figure()
#    plt.pcolormesh(seg_time, freq, spectrogram, cmap="viridis", norm=LogNorm())
    return spectrogram.T, seg_time, freq
    
def rolling(data, window=3):
    '''
    Param:
    data: array-like data
    window: int, number of frames to stack together to predict future
    noverlap: int, how many frames overlap with last window
    Return:
    rolling_data: array like '''
    
    if len(data.shape) < 2:
        shape = (data.size - window + 1, window)
        strides = (data.itemsize, data.itemsize)
    else:
        shape = (data.shape[0] - window + 1, window*data.shape[1])
        strides = (data.itemsize*data.shape[1], data.itemsize)
    
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

