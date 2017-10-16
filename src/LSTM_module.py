import sys
sys.path.append("../src/")
import modules as mod
import get_mnist as gm
import tensorflow as tf
import numpy as np


class LSTM_Cell(mod.ComposedModule):
    '''Define LSTM net'''
    def define_inner_modules(self, name, activations, input):
        '''Define LSTM cells
        three gates
        cell_state
        output
        input: 2 inputs--input[t] and outputs[t-1]'''
        self.gates = {}
        self.eleMulti = {}
        # activation functions
        activations = [tf.sigmoid, tf.tanh]
        # concate input
        self.inputconcat = mod.ConcatModule("catIn")            # make!
        # forget gate layer
        self.gates["fg"] = mod.LSTMGateModule("fg", acitvations[0])   # make!
        self.gates["ig"] = mod.LSTMGateModule("fg", acitvations[0])
        self.gates["og"] = mod.LSTMGateModule("fg", acitvations[0])
        
        self.csAdd["csAdd"] = mod.AddModule("csAdd", activations[1])   # make!
        
        self.eleMulti["multi_in"] = mod.EleMultiModule("multi_in")  # element wise multiplication for input, output, and cellstate  
        self.eleMulti["multi_f"] = mod.EleMultiModule("multi_f")   # make!
        self.eleMulti["multi_out"] = mod.EleMultiModule("multi_out")
        
        self.tanhModule["intanh"] = mod.TanhModule("intanh")   #??? # make!
        self.tanhModule["tanh_out"] = mod.TanhModule("tanh_out")
        # connections
        self.inputconcat.add_input(input, hidden_pre)
        
        self.gates["fg"].add_input(self.inputconcat)
        self.eleMulti["multi_f"].add_input(self.gates["fg"], self.LSTMStateModule["cs"][-1])
        
        self.gates["ig"].add_input(self.inputconcat)
        self.tanhModule["tanh_in"].add_input(self.inputconcat)
        self.eleMulti["multi_in"].add_input(self.gates["ig"], self.tanhModule["tanh_in"])
        self.csAdd["csAdd"].add_input(self.eleMulti["multi_in"], self.eleMulti["multi_f"])

        self.gates["og"].add_input(self.inputconcat)
        self.tanhModule["tanh_out"].add_input(self.csAdd["csAdd"])
        self.eleMulti["multi_out"].add_input(self.gates["og"], self.tanhModule["tanh_out"])
        # set input and output
        self.input_module = self.inputconcat
        self.output_module = (self.eleMulti["multi_out"], self.csAdd["csAdd"])

  
        

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
    def operation(self, x, y):
        return tf.multiply(x, y, name=self.name)

class ConcatModule(OperationModule):
    # Return concatenated vector
    def operation(self, x, y):
        # x-column vector n*1
        # y-column cector m*1
        return tf.concat([x, y], 1, name="concat_in")  # stack them vertically

class LSTMStateModule(ComposedModule):
    '''THis is a LSTM state module whose inputs are
                                1. fg (eleMulti) cell_state_pre
                                2. ig
                                3. x(t), h(t-1)
    and output is eleAdd of 1 and 2
    '''
    def define_inner_modules(self, name, activation):
        #self.input_module = 1, 2, concat(3)
        
        self.Tanh = ActivationModule(name + "_tanh", tf.tanh)
        self.EleMulti = EleMultiModule(name + "_eleMulti")
        self.bias = BiasModule(name + "_bias", (1, n+m))
        self.weights = tf.Variable(tf.truncated_normal(shape=(n, n+m), stddev=0.1), name=name+"weights")

        self.Tanh.add_input(self.bias)
        self.EleMulti.add_input(self.Tanh)

        self.input_module = self.Tanh
        self.output_module = AddModule(name + "_Add")
        self.output_module.add_input(self.EleMulti)
        
        
      
