
from Base import BaseLayer
import numpy as np

class TanH(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.forward_value=None

    def forward(self, input_tensor):
        self.forward_value = (np.exp(input_tensor) - np.exp(-1*input_tensor))/(np.exp(input_tensor) + np.exp(-1*input_tensor))
        super().forward(self.forward_value)

        return self.forward_value


    def backward(self, error_tensor):
        super().backward(error_tensor)
        return error_tensor * (1-self._input_tensor.pop()**2)