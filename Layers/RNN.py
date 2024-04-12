from audioop import error
import sys
sys.path.append('Layers')


import numpy as np
from Base import BaseLayer
from Initializers import UniformRandom,Constant
from TanH import TanH
from Sigmoid import Sigmoid
from FullyConnected import FullyConnected
from copy import deepcopy

class RNN(BaseLayer):
    def __init__(self,input_size, hidden_size, output_size)-> None:

        self.xtilda = FullyConnected(input_size+hidden_size , hidden_size)
        self.why =FullyConnected(hidden_size,output_size)
        super().__init__()
        self.trainable = True
        unifrom = UniformRandom()
        constant = Constant(0)
        self.tanh=TanH()
        self.sigmoid = Sigmoid()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size


        self.memorize = False    
        #self.whh=FullyConnected(hidden_size,hidden_size,False)
        #self.wxh=FullyConnected(input_size,hidden_size)

        self.hidden_vector = None

        self._optimizer=None
        self._gradient_weights = 0
        self.initialize(unifrom,constant)

        

    
    def _forward(self,input_tensor):

        if self.memorize is False or self.hidden_vector is None:
            self.hidden_vector =  Constant(0).initialize((input_tensor.shape[1],self.hidden_size))#b,h
        out = np.zeros((input_tensor.shape[0], input_tensor.shape[1] , self.output_size))#t,b,x\
        o = 0
        for t in range (input_tensor.shape[0]):
            oxtilda = self.xtilda.forward(np.concatenate([input_tensor[t] , self.hidden_vector],axis= 1  ))
            self.hidden_vector = self.tanh.forward(  oxtilda  )#b ,h
            out[t] = self.sigmoid.forward(  self.why.forward(self.hidden_vector) )
            #o = self.why.forward(self.hidden_vector)
            #self.sigmoid.forward(  o )

        
        #out[-1] = o#self.sigmoid.forward(  self.why.forward(self.hidden_vector) )


        return out
    
    def forward(self, input_tensor):
        super().forward(input_tensor) # S ( tanh(t,b,x  * x,h + b,h * h,h + bias) * h,y + bias )= output (b,y)
        if len(input_tensor.shape) == 2:
            input_tensor = np.expand_dims(input_tensor,1)
            o = self._forward(input_tensor)[:,0,:]
            return o
        else :
            o = self._forward(input_tensor)
            return o 

    def _backward():
        pass
    def backward(self, error_tensor):
        inp = self._input_tensor.pop()
        
        error_tensor = np.expand_dims(error_tensor , 1)
        gradh =  np.zeros((error_tensor.shape[1],self.hidden_size))
        gradx = np.empty_like(inp)

        weight_1 = 0
        weight_2 = 0

        for t in range(error_tensor.shape[0]-1 , -1 , -1):
            

            e = self.sigmoid.backward(error_tensor[t])
            e = self.why.backward(e,False)
            weight_2 += self.why.gradient_weights
            e = e + gradh
            e = self.tanh.backward(e)
            e = self.xtilda.backward(e,False)
            weight_1 +=  self.xtilda.gradient_weights

            gradh = e[:,self.input_size:]
            gradx[t] = e[:,:self.input_size]

        self.xtilda.update_weights(weight_1)
        self.why.update_weights(weight_2)


        """
        [[0.00701419 0.00752896 0.0076501  0.00932856 0.01095827]
 [0.00697803 0.00749661 0.00760005 0.00928499 0.01091825]
 [0.00697807 0.00749664 0.0076001  0.00928504 0.0109183 ]
 [0.00697827 0.00749673 0.00760035 0.0092852  0.01091845]
 [0.00697801 0.00749662 0.00760004 0.00928501 0.01091826]
 [0.00697832 0.00749677 0.00760045 0.00928527 0.01091851]
 [0.00697802 0.00749661 0.00760005 0.009285   0.01091825]
 [0.00697806 0.00749662 0.00760008 0.00928501 0.01091827]
 [0.00698005 0.00749751 0.00760244 0.00928642 0.0109198 ]]"""

        #print('shape:',self.gradient_weights.shape)

        

        
        self.gradient_weights = weight_1
        return np.array(gradx)
        pass


       
    def initialize(self,weights_initializer,bias_initializer):

        self.xtilda.initialize(weights_initializer,bias_initializer)
        self.why.initialize( weights_initializer , bias_initializer) 



    @property
    def gradient_weights(self,):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self,a):
        self._gradient_weights = a 

    @property
    def weights(self,):
        return self.xtilda.weights
        pass
     

    @weights.setter
    def weights(self,a):
        self.xtilda.weights = a
        pass
    

    @property
    def optimizer(self,):
        return self._optimizer
        pass

    @optimizer.setter
    def optimizer(self,op):
        self._optimizer = op
        self.xtilda.optimizer = self._optimizer
        self.why.optimizer = deepcopy(self._optimizer)
        pass

