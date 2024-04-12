
from abc import ABC,abstractmethod

class BaseLayer(ABC):
    pass
class BaseLayer(ABC):
    def __init__(self) -> None:
        self.trainable=False    
        self.weights=None
        self._input_tensor = []
        self.testing_phase= False   

        pass
    def __call__(self,l:BaseLayer)->BaseLayer:
        pass
    @abstractmethod
    def forward(self,input_tensor):
        if self.testing_phase==False:
            self._input_tensor.append(input_tensor)
        pass
    
    @abstractmethod
    def backward(self,error_tensor): 
        if  len(self._input_tensor )==0 and self.trainable==True:
            raise Exception('No forward called!')
        pass
    
 
        
