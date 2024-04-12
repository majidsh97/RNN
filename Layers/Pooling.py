from Base import BaseLayer
import numpy as np
import scipy

class Pooling(BaseLayer):
    def __init__(self,stride_shape, pooling_shape) -> None:
        super().__init__()

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape 
        self._bc = []


    def forward(self, input_tensor):
        super().forward(input_tensor)


        o = np.zeros( ( input_tensor.shape[0]  , input_tensor.shape[1] ,int(( input_tensor.shape[2]-self.pooling_shape[0] ) / self.stride_shape[0] ) +1 ,int( ( input_tensor.shape[3]-self.pooling_shape[1]  ) / self.stride_shape[1] )+1 )  ) 
        self._bc = np.empty((o.shape[2],o.shape[3]),dtype=object)
        for j in range(o.shape[2]):
            for i in range(o.shape[3]):
                w = input_tensor[:,:, j * self.stride_shape[0] : j * self.stride_shape[0]+self.pooling_shape[0]  ,i * self.stride_shape[1] : i * self.stride_shape[1] + self.pooling_shape[1] ]
                am = np.amax(w,(2,3),keepdims=True)

                o[:,:,j,i]=am[:,:,0,0]

                pos = np.argwhere(w==am)
                pos[:,3] += i * self.stride_shape[1]
                pos[:,2] += j * self.stride_shape[0]

                self._bc[j,i]=pos
                

        self._bc = np.array(self._bc)
                



        return o
        

    def backward(self, error_tensor):
         super().backward(error_tensor)
         #print('back',error_tensor.shape,self._bc.shape)
         o = np.zeros_like(self._input_tensor)
         for j in range(error_tensor.shape[2]):
            for i in range(error_tensor.shape[3]):
                for p in self._bc[j,i]:
                    o[p[0], p[1], p[2], p[3]] += error_tensor[p[0],p[1],j,i]


         return o

   
"""
t = np.random.uniform(0,1,size = (32,3,10,10) )
print(t.shape)
p = Pooling((1,1),(2,2))
f =p.forward(t)
print(f.shape)
#print(p._bc)
b= p.backward(f)
print(b.shape)
"""
