import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
params = {'legend.fontsize': 16,
          'figure.figsize': (10, 8.8),
         'axes.labelsize': 20,
         #'weight' : 'bold',
         'axes.titlesize':20,
         'xtick.labelsize':16,
         'ytick.labelsize':16}
pylab.rcParams.update(params)
import matplotlib

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
        plt.plot(prediction, 'y-', label="prediction",  alpha=0.5)
        plt.legend(loc="best")
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        if ifsave:
            plt.savefig(save_name+".png", format="png")

    def plot_score(self, score, epochs, title="Score on training epochs", x="epoch", y="R^2 score", save_name="score in training epochs"):
        plt.plot(np.arange(epochs)+1, np.array(score), "bo-")
        plt.title(title)
        plt.ylabel(y)
        plt.xlabel(x)
        plt.ylim([0, 1.0])
        plt.savefig(save_name, format="png")
        
