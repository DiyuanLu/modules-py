import sys
sys.path.append("../src/")
import modules as mod
import get_mnist as gm
import tensorflow as tf
import numpy as np


class LSTM_Cell(mod.ComposedModule):
    '''Define LSTM net'''
    def __init__(self, name, cell_size):
        self.cell_size = cell_size
    def define_inner_modules(self, name, in_size, self.cell_size):
        """Typical LSTM cell with three gates. Detailed tutorial see
        http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
        """
        self.input_module = ConcatModule("concat", 1)
        # Three gates for input, output, cell state
        f_t = mod.FullyConnectedLayerModule("f_t", tf.sigmoid, self.cell_size, in_size+self.cell_size)
        i_t = mod.FullyConnectedLayerModule("i_t", tf.sigmoid, self.cell_size, in_size+cell_size)
        o_t = mod.FullyConnectedLayerModule("o_t", tf.sigmoid, self.cell_size, in_size+self.cell_size)
        
        # transformed input and last time hidden-state
        CHat_t = mod.FullyConnectedLayerModule("CHat_t", tf.tanh, self.cell_size, in_size+self.cell_size)
        # current cell state
        self.C_t = mod.AddModule("C_t")
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

        state_update.add_input(i_t)
        state_update.add_input(CHat_t)

        self.C_t.add_input(state_update)  # C_t = state_redidual + state_update
        self.C_t.add_input(state_residual)

        tanh_C_t.add_input(self.C_t)
        self.h_t.add_input(o_t)   
        self.h_t.add_input(tanh_C_t)   
        
        # set input and output
        self.input_module = self.inputconcat
        self.output_module = self.h_t

  
        

class LSTMGateModule(ComposedModule):
    ''' THis is the gates module for LSTM cells
    Param;s
    name
    activations: gates activation function is tf.sigmoid
    inputs; gates module take two inputs: i[t], o[t-1]
    Output;
    gate_out: float 0~1'''
    def define_inner_modules(self, name, activation):
        self.weights = tf.Variable(tf.truncated_normal(shape=(n, n+m), stddev=0.1), name=name)   #  ?????
        self.bias = BiasModule(name + "_bias", (1, out_size))
        self.preactivation = AddModule(name + "_preactivation")
        self.output_module = ActivationModule(name + "_output", activation)
        self.preactivation.add_input(self.input_module)
        self.preactivation.add_input(self.bias)
        self.preactivation.add_input(self.weights)
        self.output_module.add_input(self.preactivation)

class EleMultiModule(OperationModule):
    ## Returns element wise multiplication
    def operation(self, *args):
        x = args[0]
        for y in args[1:]:
            x = tf.multiply(x, y, name=self.name)
        return x

class ConcatModule(OperationModule):
    def __init__(self, name, axis):
        super().__init__(self, name, axis)
        self.axis = axis
        
    # Return concatenated vector
    def operation(self, *args):
        return tf.concat(args, self.axis, name=self.name)  # stack them vertically


        
        
      
