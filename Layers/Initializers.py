from abc import ABC,abstractmethod
import numpy as np

class BaseInit(ABC):
    
    def __init__(self) -> None:
        self.weights_shape = None
        self.fan_in = None 
        self.fan_out=None
        pass

    @abstractmethod
    def initialize(self, weights_shape,fan_in=None, fan_out=None):
        self.fan_in=fan_in
        self.fan_out=fan_out
        self.weights_shape = weights_shape
        pass


class Constant(BaseInit):
    def __init__(self,constant=0.1) -> None:
        super().__init__()
        self.constant = constant
    pass

    def initialize(self,weights_shape, fan_in=None, fan_out=None):
         super().initialize(weights_shape,fan_in, fan_out)
         w = np.ones(weights_shape)*self.constant
         
         
         return w
         



class UniformRandom(BaseInit):
    def __init__(self) -> None:
        super().__init__()
    pass

    def initialize(self,weights_shape, fan_in=None, fan_out=None):
         super().initialize(weights_shape, fan_in, fan_out)
         w = np.random.uniform(0,1,weights_shape)
         return w

   
class Xavier(BaseInit):
    def __init__(self) -> None:
        super().__init__()
    pass

    def initialize(self,weights_shape, fan_in, fan_out):
         super().initialize(weights_shape,fan_in, fan_out)
         sigma = np.sqrt(2)/np.sqrt(fan_in+fan_out)
         
         return np.random.normal(0,sigma,weights_shape)
    
class He(BaseInit):
    def __init__(self) -> None:
        super().__init__()
    pass

    def initialize(self,weights_shape, fan_in, fan_out):
         super().initialize(weights_shape,fan_in, fan_out)
         sigma = np.sqrt(2/fan_in)
         return np.random.normal(0,sigma,weights_shape)