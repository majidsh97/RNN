import sys

sys.path.append('Layers')

from Base import BaseLayer
from scipy.signal import convolve,correlate
from Initializers import UniformRandom,Constant
import numpy as np
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels) -> None:
        super().__init__()
        self.trainable=True

        self.stride_shape = stride_shape
        self.sy = self.stride_shape[0]
        self.sx = 1
        if len(stride_shape)>1:
            self.sx=self.stride_shape[1]
        self.c = convolution_shape[0]
        self.y = convolution_shape[1]
        self.x = 1
        if len(convolution_shape)>2:
            self.x = convolution_shape[2]

        self.num_kernels = num_kernels
        self.forward_value = None
        self._input_shape=None
        self.gradient_bias=None
        self.gradient_weights=None
        self._optimizer=None
        self._optimizer_b=None
        self.weights=None
        self.bias=None
        self.initialize(UniformRandom(),UniformRandom())#Constant(0))
        #self.optimizer = Sgd(0.001)

        pass
    def initialize(self,w,b):
        self.weights=w.initialize((self.num_kernels, self.c, self.y, self.x),self.c*self.x*self.y,self.num_kernels*self.x*self.y)
        self.bias= b.initialize((self.num_kernels,1),self.c*self.x*self.y,self.num_kernels*self.x*self.y)


    def _pad(self,inp,filter_y,filter_x):
        padx = int((filter_x - 1) / 2)
        pady = int((filter_y - 1) / 2)
        _px=0
        _py=0

        if filter_x%2==0:
            _px=1
        if filter_y%2==0:
            _py=1

        pad_with=((0,0),(0,0), (pady+_py,pady), (padx+_px,padx))
        
        padded = np.pad(
                inp, pad_with, "constant", constant_values=0
            )
        return padded
    def _forward2d(self,input_tensor):
        #self._input_tensor=input_tensor
        self._input_shape=input_tensor.shape
        output = np.zeros(
            (
                input_tensor.shape[0],
                self.num_kernels,
                input_tensor.shape[2],
                input_tensor.shape[3],
            )
        )
        input_tensor = self._pad(input_tensor, self.y,self.x)
        self._input_tensor = input_tensor

        for j in range(input_tensor.shape[0]):
            for i in range(self.num_kernels):
                o = correlate(input_tensor[j], self.weights[i], mode="valid")
                output[j, i] = o + self.bias[i]

        self._oshape = output.shape
        output=output[:,:,::self.sy,::self.sx]

        return output

    def forward(self, input_tensor):
        #print(input_tensor.shape)
        #b c y x
        if len(input_tensor.shape)==3:
            input_tensor= input_tensor[:,:,:,np.newaxis]
            output = self._forward2d(input_tensor)[:,:,:,0]
            return output
        else:
            output = self._forward2d(input_tensor)
            return output
       
        pass

    def _backward2d(self,error_tensor):
        input_tensor = self._input_tensor

        ez = np.zeros(self._oshape)
        ez[:,:,::self.sy,::self.sx] = error_tensor
        error_tensor = ez.copy() 

        dydw = np.zeros_like(self.weights)
        dydb = np.zeros_like(self.bias)
        dydx = np.zeros(self._input_shape)
        self.gradient_weights = dydw.copy()
        self.gradient_bias = dydb.copy()

        wflip=np.flip(self.weights,0)
        err_padded =  self._pad(error_tensor,self.weights.shape[2],self.weights.shape[3])


        for j in range(input_tensor.shape[0]):
            for i in range(self.num_kernels):
                dydw[i] = correlate(input_tensor[j], error_tensor[j,i:i+1], mode="valid")
                dydb[i] = np.sum(error_tensor[j,i:i+1]) #convolve(1, error_tensor[j,i:i+1], mode="valid")
               

            self.gradient_weights += dydw 
            self.gradient_bias += dydb

            for ic in range(self.c):
                o = convolve(err_padded[j], wflip[:,ic], mode="valid")#[:,1:-1,1:-1]
                dydx[j,ic]=o


        if self._optimizer is not None:
            self.weights =  self._optimizer.calculate_update(self.weights,self.gradient_weights)
            self.bias =  self._optimizer_b.calculate_update(self.bias,self.gradient_bias )
            pass   

        return dydx

    def backward(self, error_tensor):

        if len(error_tensor.shape)==3:
            error_tensor= error_tensor[:,:,:,np.newaxis]
            output = self._backward2d(error_tensor)[:,:,:,0]
            return output
        else:
            output = self._backward2d(error_tensor)
            return output

    @property
    def optimizer(self,):
        return self._optimizer;


    pass

    @optimizer.setter
    def optimizer(self,op):
        self._optimizer=op
        self._optimizer_b =  copy.deepcopy(op)

