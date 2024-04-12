import sys
sys.path.append('Layers')
from Base import BaseLayer
import numpy as np
from ReLU import ReLU
from Initializers import UniformRandom,Constant

class FullyConnected(BaseLayer):
    def __init__(self,input_size, output_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable=True
        self._optimizer=None
        self.weights= None #np.random.uniform(0,1,(input_size+1,output_size )) 
        self.initialize(UniformRandom(),Constant())
        #rint(self.weights.shape)
        self.gradient_weights=None
        self.__forward_out=None



    def forward(self, input_tensor):
        
        input_tensor =np.concatenate([input_tensor,np.ones((input_tensor.shape[0],1))],1) #wx +b , w*x
        super().forward(input_tensor)
        self.__forward_out = np.matmul( input_tensor , self.weights )  # (m x inp) X (inp x out ) -> (m x out) + (m *x out)
        #print(o.shape)
        return  self.__forward_out

    def backward(self,error_tensor,update=True): # (m x out) (mxinp) -> (inp x out)
        super().backward(error_tensor)
        xt=np.transpose( self._input_tensor.pop() )


        new_error = np.matmul( xt,error_tensor )
        self.gradient_weights = new_error

        if update:
            self.update_weights(self.gradient_weights)

        #if self._optimizer is not None:
        #    self.weights =  self._optimizer.calculate_update(self.weights,new_error)

        
        return np.matmul( error_tensor,self.weights[:-1,:].T ) # m x out out x inp -> m * inp
    def update_weights(self,gw):
        if self._optimizer is not None:
            self.weights =  self._optimizer.calculate_update(self.weights,gw)
        pass


    @property #-> self.optimizer=2 raise error
    def optimizer(self,):
        return self._optimizer

    @optimizer.setter # self.optimizer = 2 no error
    def optimizer(self,value):
        self._optimizer=value

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size,self.output_size ),fan_in=self.input_size, fan_out=self.output_size)
        b = bias_initializer.initialize((1,self.output_size ),fan_in=1, fan_out=self.output_size) 
        self.weights = np.concatenate([self.weights, b] ,axis=0)
        




