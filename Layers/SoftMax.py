from Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.__softmax_out=None
    
    def forward(self, input_tensor):
         super().forward(input_tensor)
         
         e = np.exp(input_tensor-np.max(input_tensor))
         s = np.sum(e,-1,keepdims=True)
        
         self.__softmax_out= np.divide(e,s)
         return self.__softmax_out
         

    def backward(self, error_tensor):
        super().backward(error_tensor)
        e = self.__softmax_out *( error_tensor-np.sum(error_tensor *self.__softmax_out ,axis=1,keepdims=True))
        

        return e




