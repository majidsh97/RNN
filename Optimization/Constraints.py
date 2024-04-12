import numpy as np

class L2_Regularizer():

    def __init__(self,alpha) -> None:
        self.alpha = alpha
        pass   

    def calculate_gradient(self,weights):
        return self.alpha * weights
        pass

    def norm(self,weights):
        return self.alpha * np.sum(weights**2)
        pass
    
class L1_Regularizer():

    def __init__(self,alpha) -> None:
        self.alpha = alpha
        pass

    def calculate_gradient(self,weights):
        return self.alpha * np.sign( weights )

        pass

    def norm(self,weights):
        return self.alpha * np.sum(np.abs( weights))
        pass
