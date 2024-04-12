from Base import BaseLayer
import numpy as np
class Flatten(BaseLayer):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, input_tensor ):
         super().forward(input_tensor)
         return np.reshape(input_tensor,(input_tensor.shape[0],-1))


    def backward(self, error_tensor):
         super().backward(error_tensor)
         return np.reshape(error_tensor,self._input_tensor.pop().shape)
    pass