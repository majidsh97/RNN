from Base import BaseLayer
import random
import numpy as np

class Dropout(BaseLayer):

    def __init__(self,probability) -> None:
        super().__init__()
        self.probability = probability
        

    def forward(self, input_tensor):

        if  self.testing_phase==False  :
            #print('shape:',input_tensor.shape)
            #rc = random.choices([1/self.probability,0],[self.probability,1-self.probability],k=input_tensor.shape[1])
            rc = np.random.binomial(1,self.probability,input_tensor.shape) * 1/self.probability
            super().forward(rc)
            
            return  input_tensor * rc
        else :
            return input_tensor



    def backward(self, error_tensor):
        super().backward(error_tensor)
        error_tensor *=  self._input_tensor.pop()
        return error_tensor #* 1/self.probability

    pass



