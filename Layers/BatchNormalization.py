from Base import BaseLayer
import numpy as np
from Initializers import Constant
from Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):

    def __init__(self,channels) -> None:
        super().__init__()
        
        self.trainable=True
        self.mu=0
        self.sigma=0
        self.weights = None
        self.bias = None

        self.channels = channels
        self._optimizer=None
        self.mu_tilda = None
        self.sigma_tilda = None
        self.alpha = 0.8
        self.__input_shape = None
        self.__xtilda_back = []
        self.gradient_bias=None
        self.gradient_weights = None

        self.initialize()

    def _forward(self, input_tensor):

        if self.testing_phase:
            xtilda = (input_tensor - self.mu_tilda)/(np.sqrt(self.sigma_tilda**2+np.finfo(float).eps))
            o = self.weights * xtilda + self.bias
            return o

        else :
            super().forward(input_tensor)

            self.mu = np.mean(input_tensor,0,keepdims=True)
            self.sigma = np.std(input_tensor,0,keepdims=True)

            xtilda = (input_tensor - self.mu)/(np.sqrt(self.sigma**2+np.finfo(float).eps))
            self.__xtilda_back.append(xtilda)

            if self.mu_tilda is None or self.sigma_tilda is None :
                self.mu_tilda = self.mu
                self.sigma_tilda = self.sigma
            else :
                self.mu_tilda = self.alpha * self.mu_tilda + (1 - self.alpha) * self.mu
                self.sigma_tilda = self.alpha * self.sigma_tilda + (1 - self.alpha) * self.sigma

            o = self.weights * xtilda + self.bias
            return o
        

    def forward(self, input_tensor):
        #self.__input_shape = input_tensor.shape
        if len(input_tensor.shape)==4:
             inp = self.reformat(input_tensor)
             o = self._forward(inp)
             o = self.reformat(o)
             return o

        else:
            return self._forward(input_tensor)

    def _backward(self, error_tensor):
        xtilda = self.__xtilda_back.pop()
        self.gradient_bias = np.sum(error_tensor,0,keepdims=True)
        self.gradient_weights = np.sum(error_tensor*xtilda,0,keepdims=True)
        if self._optimizer is not None:
            self.weights =  self._optimizer.calculate_update(self.weights,self.gradient_weights)
            self.bias =  self._optimizer.calculate_update(self.bias,self.gradient_bias)
        pass
        
        dldx = compute_bn_gradients(error_tensor,self._input_tensor.pop(),self.weights,self.mu,self.sigma**2)
        return dldx
        

        
    def reformat(self,tensor):
       if len(tensor.shape)==4:
           self.__input_shape = tensor.shape

           tensor = np.reshape(tensor,(tensor.shape[0],tensor.shape[1] ,-1))
           tensor = np.transpose(tensor,[0,2,1])
           tensor = np.reshape(tensor,(-1,tensor.shape[2]))
           return tensor
       else:
           shape = self.__input_shape
           tensor = np.reshape(tensor,(shape[0],shape[2]*shape[3],shape[1]))
           tensor = np.transpose(tensor, [0,2,1])
           tensor = np.reshape(tensor, (shape[0],shape[1],shape[2],shape[3]))
           return tensor






    def backward(self, error_tensor):
        super().backward(error_tensor)

        if len(error_tensor.shape)==4:
             err = self.reformat(error_tensor)
             o = self._backward(err)

             o = self.reformat(o)
             return o

        else:
            return self._backward(error_tensor)
    pass

    def initialize(self,wi=None , bi=None):
        self.weights= Constant(1).initialize((1,self.channels))
        self.bias= Constant(0).initialize((1,self.channels))
        pass

    @property #-> self.optimizer=2 raise error
    def optimizer(self,):
        return self._optimizer

    @optimizer.setter # self.optimizer = 2 no error
    def optimizer(self,value):
        self._optimizer=value


