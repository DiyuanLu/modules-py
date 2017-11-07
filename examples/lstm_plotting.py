import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import ipdb

params = {'legend.fontsize': 16,
          'figure.figsize': (10, 8.8),
         'axes.labelsize': 20,
         #'weight' : 'bold',
         'axes.titlesize':20,
         'xtick.labelsize':16,
         'ytick.labelsize':16}
pylab.rcParams.update(params)
import matplotlib

def get_autocorr(data):
    '''
    Param:
    data: array-like data'''
    err = data - np.mean(data)
    norm = np.sum(err**2)
    temp = np.correlate(err, err, mode='full')/norm
    result = temp[temp.size/2:]

    return result
class Plot(object):

    def plot_compare_threshold(self, target, data, title="compare threshold"):
        '''
        target: 7*frames
        data: 7*frames'''
        plt.figure(title+np.str(threshold)+"threshold")
        plt.subplot(311)
        plt.plot(target, interpolation="nearest", cmap="Blues")
        plt.title(title+" target")
        plt.xlabel("steps")
        plt.subplot(312)
        plt.plot(data , interpolation="nearest", cmap="Blues")
        plt.title(title+" original")
        plt.xlabel("steps")
        plt.colorbar(orientation='horizontal')
        #plt.subplot(313)
        #plt.imshow((data.T>=np.mean(data, axis=1))[-self.par.seq_length*3:, :].T+0 , interpolation="nearest", cmap="Blues", aspect="auto")
        #assert np.mean(data, axis=1).shape[0] == 7
        #plt.title(title+"mean threshold")
        #plt.xlabel("steps")
        #plt.colorbar()
        plt.subplot(313)
        plt.imshow((data[:, -self.par.seq_length*3:]>=threshold)+0 , interpolation="nearest", cmap="Blues")
        plt.title(title+" {} threshold".format(threshold))
        plt.xlabel("steps")
        plt.tight_layout()
        plt.savefig(title+np.str(threshold)+"threshold-", format="png")
        plt.close()


    def plotimshow(self, matrix, cmap="Blues", title="plot matrix", x="steps", y="value"):
        '''matrix: 2D array-N_e*steps'''
        plt.imshow(matrix, interpolation="nearest", cmap=cmap, aspect="auto")
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.colorbar()


    def plot_compare(self, target, prediction, title="plot compare audio", x="time", y="normalized amplitude", save_name="prediction_sorn", ifsave=True):
        plt.plot(target, 'b-', label='target', lw=1)
        plt.hold(True)
        plt.plot(prediction, 'c-', label="prediction", alpha=0.6)   # ,  
        plt.legend(loc="best")
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        if ifsave:
            plt.savefig(save_name+".png", format="png")
            plt.close()

    def plot_score_differ_epochs(self, score, epochs, title="Score on training epochs", x="epochs", y="R^2 score", save_name="score in training epochs", ifsave=True):
        plt.plot(score, "c*-")
        plt.xticks(np.arange(epochs), np.arange(epochs)+1)
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)
        
        #plt.ylim([0, 1.0])
        if ifsave:
            plt.savefig(save_name, format="png")
            plt.close()

    def plot_score_differ_delays(self, score, epochs, max_delay, title="Score on training epochs", x="frame delay", y="R^2 score", save_name="score in training epochs", ifsave=True):
        if epochs == 0:
            epochs += 1
        plt.plot(score, "go-")
        plt.xticks(np.arange(epochs*max_delay), np.tile(np.arange(1,max_delay+1), epochs))
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)
        
        if ifsave:
            plt.savefig(save_name, format="png")
            plt.close()

    def plot_score(self, score, title="MSE on training epochs", x="epochs", y="MSE", save_name="MSE in training epochs", ifsave=True):
        plt.plot(score, "go-")
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)
        
        if ifsave:
            plt.savefig(save_name, format="png")
            plt.close()

    def plot_autocorr(self, x, title="Auto correlation", save_name="Autocorrelation", ifsave=False):
        '''
        Param:
        x: array-like data'''
        result = get_autocorr(x)

        plt.figure()
        plt.subplot(211)
        plt.plot(result, 'b*-')
        plt.title(title+"zoomed out")
        plt.xlabel("delays")
        plt.ylabel("autocorrelation")
        plt.xlim([-2, len(x)])
        plt.subplot(212)
        plt.plot(result[0:200], 'm*-')
        plt.title(title+"zoomed in")
        plt.xlabel("delays")
        plt.ylabel("correlation")
        plt.xlim([-2, 200])
        if ifsave:
            plt.savefig(save_name+".png", format="png")
            plt.close()
      
    def plot_specgram_and_wav(self, frames, sampFreq, cmap="viridis", title="Spectrogram", save_name="spectrogram", ifsave=False):
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.05, right=0.051)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.7, 1.3])
        ax = plt.subplot(gs[0])
        cmap = plt.get_cmap(cmap) #'viridis' this may fail on older versions of matplotlib
        vmin = -40  # hide anything below -40 dB
        cmap.set_under(color='k', alpha=None)

        #fig, ax = plt.subplots()
        if len(frames.shape) != 1:
            frames = frames[:, 0]     # first channel
        pxx, freq, t, cax = ax.specgram(frames, 
                                        Fs=sampFreq,      # to get frequency axis in Hz
                                        cmap=cmap, vmin=vmin)
        #cbar = fig.colorbar(cax)
        #cbar.set_label('Intensity dB')
        ax.axis("tight")

        # Prettify
        import matplotlib
        import datetime

        ax.set_xlabel('time/s')
        ax.set_ylabel('frequency kHz')

        scale = 1e3                     # KHz
        ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
        ax.yaxis.set_major_formatter(ticks)
        
        def timeTicks(x, pos):
            d = datetime.timedelta(seconds=x)
            return str(d.seconds)    # original d: get hour, min, seconds
        
        formatter = matplotlib.ticker.FuncFormatter(timeTicks)
        ax.xaxis.set_major_formatter(formatter)
        plt.subplot(gs[1])
        plt.plot((frames / (2.**15)), 'c-')
        plt.xlim([0, len(frames)])
        plt.ylabel("amplitude")
        plt.xlabel("frames")
        plt.tight_layout()

        if ifsave:
            plt.savefig(save_name+".png", format="png")
            plt.close()

    def reconstruct_audio_from_spec(self, data):
        '''
        Param:
        data: array-like data, wav data'''
        
        dt = 0.1  #  define a time increment (seconds per sample)
        N = len(data) 

        Nyquist = 1/(2*dt)  #  Define Nyquist frequency
        df = 1 / (N*dt)  #  Define the frequency increment

        G = np.fft.fftshift(np.fft.fft(data))  #  Convert "data" into frequency domain and shift to frequency range below
        f = np.arange(-Nyquist, Nyquist-df, df) #  define frequency range for "G"
        ipdb.set_trace()
        if len(G) != len(f):
            length = min(len(G), len(f))
        G_new = G[:length]*(1j*2*np.pi*f[:length]) 

        data_rec = np.abs(np.fft.ifft(np.fft.ifftshift(G_new)))

        plt.figure()
        plt.plot(data, 'b-', label='original')
        plt.hold(True)
        plt.plot(data_rec, 'm-', label='reconstruction', alpha=0.6)
        plt.legend(loc="best")
        plt.xlabel("frames")
        #plt.xlim([-2, 200])

    def plot_compare_specgram(self,seg_time1, freqs1, spec1, tar_seg_time, tar_freqs, target_spec, save_name="compare spectrogram", ifsave=True):
        '''
        Param:
            spec1: 2D array network predicted spectrogram
            target_spec: 2D array target spectrogram'''

        plt.figure()
        plt.subplot(211)
        plt.pcolormesh(tar_seg_time, tar_freqs, 10*np.log10(np.abs(target_spec)), cmap="viridis")   #, norm=LogNorm()
        plt.title("Target spectrogram")
        plt.xlabel("time")
        plt.ylabel("frequency")
        plt.colorbar()
        plt.subplot(212)
        plt.pcolormesh(seg_time1, freqs1, 10*np.log10(np.abs(spec1)), cmap="viridis")   #, norm=LogNorm()
        plt.title("Predicted spectrogram")
        plt.xlabel("time")
        plt.ylabel("frequency")
        plt.colorbar()
        plt.tight_layout()
        if ifsave:
            plt.savefig(save_name+".png", format="png")
            plt.close()
            
        

