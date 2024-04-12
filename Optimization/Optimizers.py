import numpy as np

class Optimizer():
	def __init__(self,learning_rate=None):
		self.regularizer = None
		self.learning_rate=learning_rate

		pass
	def add_regularizer(self,regularizer):
		self.regularizer = regularizer
		pass

	def calculate_gradient(self,weights):
		if self.regularizer is not None:
			return -self.learning_rate * self.regularizer.calculate_gradient(weights)
		else:
			return 0

		pass

class SgdWithMomentum(Optimizer):

	def __init__(self,learning_rate,momentum_rate) -> None:
		super().__init__(learning_rate)
		self.v=0	
		self.momentum_rate = momentum_rate
		pass

	def calculate_update(self,weight_tensor, gradient_tensor):
		self.v = self.momentum_rate*self.v - self.learning_rate * gradient_tensor
		new_weight = weight_tensor + self.v + self.calculate_gradient(weight_tensor)
		return new_weight

class Adam(Optimizer):
	
	def __init__(self,learning_rate,mu,rho) -> None:
		super().__init__(learning_rate)
		self.mu=mu
		self.rho=rho
		self.v=0
		self.r=0
		self.t=1
		
		pass

	def calculate_update(self,weight_tensor, gradient_tensor):
		self.v = self.mu * self.v + (1-self.mu) * gradient_tensor
		self.r = self.rho * self.r + (1-self.rho)*gradient_tensor **2
		vhat = self.v/(1- self.mu**self.t)
		rhat=self.r/(1-self.rho**self.t)
		self.t+=1
		new_weight = weight_tensor - self.learning_rate * vhat/(np.sqrt(rhat) + 1e-8)  + self.calculate_gradient(weight_tensor) #2.22044604925e-16)  
		return new_weight
	pass



class Sgd(Optimizer):

	def __init__(self,learning_rate:float) -> None:
		super().__init__(learning_rate)
		pass

	def calculate_update(self,weight_tensor, gradient_tensor):
		new_weight = weight_tensor - gradient_tensor * self.learning_rate + self.calculate_gradient(weight_tensor)
		return new_weight