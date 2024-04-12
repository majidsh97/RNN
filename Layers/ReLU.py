from Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        
        
    
    def forward(self, input_tensor):
         super().forward(input_tensor)
         input_tensor[input_tensor<0] = 0 
         return input_tensor
         

    def backward(self, error_tensor):
         super().backward(error_tensor)
         x = self._input_tensor.pop()
         x[x>0]=1
         #x[x==0]=0
         x[x<=0]=0
         return x*error_tensor





