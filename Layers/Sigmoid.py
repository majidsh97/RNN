from Base import BaseLayer
import numpy as np

class Sigmoid(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        forward_value =  1/(1+np.exp(-1*input_tensor))
        super().forward(forward_value)

        return forward_value

    def backward(self, error_tensor):
        super().backward(error_tensor)
        forward_value = self._input_tensor.pop()
        return error_tensor * forward_value*(1-forward_value)
