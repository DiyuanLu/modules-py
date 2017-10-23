import sys
sys.path.append("../src/")
import modules as mod
#import get_mnist as gm
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
