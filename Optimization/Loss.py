
import numpy as np
import math

class CrossEntropyLoss():

    def __init__(self) -> None:
        self.__prediction_tensor = None
        pass

    def forward(self,prediction_tensor, label_tensor):

        self.__prediction_tensor = prediction_tensor
        loss = -np.sum(np.multiply( label_tensor, np.log(prediction_tensor + 2.22044604925e-16)) )#,-1,keepdims=True) # 2.225e-16
       
        return loss
        pass

    def backward(self, label_tensor):
        e = (-label_tensor) /  (self.__prediction_tensor + 2.22044604925e-16)

        
        return e
        pass


