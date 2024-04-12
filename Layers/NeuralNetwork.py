#import sys;
#import os;
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))

from FullyConnected import FullyConnected
from Base import BaseLayer
from ReLU import ReLU
from SoftMax import SoftMax
from Base import BaseLayer
import copy
import numpy as np

#TODO::Refactor the NeuralNetwork class to add the regularization loss to the data loss. Use
#the method norm(weights) to get the regularization loss inside all layers (Fully Con-
#nected, Convolution and RNN) and sum it up.

class NeuralNetwork():

    def __init__(self,optimizer,weights_initializer,bias_initializer):

        self.optimizer = optimizer
        self.loss=[]
        self.layers=[]
        self.data_layer=None
        self.loss_layer=None
        self.input_tensor= None
        self.label_tensor = None#.next()
        
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        #TODO::Implement a property phase in the NeuralNetwork class setting the phase of each of
        #its layers accordingly. Use this method to set the phase in the train and test methods.
        self._phase=False


        pass

    def forward(self,):

        data  = self.data_layer.next()
        self.input_tensor=data[0]
        self.label_tensor=data[1]
        o = self.input_tensor
        regterm = [0]
        for l in self.layers:
            l.testing_phase=self.phase
            o = l.forward(o)
            
            if hasattr(l,'optimizer'):
                if l.optimizer.regularizer is not None:
                    regterm.append(l.optimizer.regularizer.norm(l.weights))

        o = self.loss_layer.forward(o,self.label_tensor) + np.sum(regterm)

        return o
        pass

    def backward(self,):
        o = self.loss_layer.backward(self.label_tensor)
        
        for i in range(len(self.layers)-1,-1,-1):
            o = self.layers[i].backward(o)
        
        return o

        pass

    def append_layer(self,layer:BaseLayer):
        if layer.trainable==True:
            op = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer,self.bias_initializer)
            layer.testing_phase = self.phase
            layer.optimizer = op

            pass

        self.layers.append(layer)
        pass

    def train(self,iterations):
        self.phase = False
        for i in range(iterations):

            o = self.forward()
            self.loss.append(o)

            self.backward()
            


    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self,phase):
        self._phase = phase
        pass

        
    def test(self,input_tensor):
        self.phase = True
        o = input_tensor
        for l in self.layers:
            l.testing_phase = self.phase
            o = l.forward(o)

        return o
        

    

