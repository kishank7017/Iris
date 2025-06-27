

import torch

i=torch.tensor(1)
print(i)

v=torch.tensor([1,2])
print(v)

v1=torch.tensor([[1,2],[3,4],[5,6]])
print(v1)

m=torch.tensor([[1,2],[3,4]])
print(m)

t=torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])
print(t)

ones1=torch.ones(size=(3,4,5))
print(ones1)

zero1=torch.zeros(size=(3,4,5))
print(zero1)

random_tensor=torch.rand(size=(2,3,5))
print(random_tensor)

# create tensor in a range
# torch.arange(start,end,step)
range_tensor=torch.arange(0,20,2.5)
print(range_tensor)

# to create a tensor similar to other in shape
similar_tensor=torch.zeros_like(range_tensor)
print(similar_tensor)

## tensor data types

## none will set it to float32 it is default
float_tensor_float32=torch.tensor([1,2,3,4],dtype=torch.float32)

## float16
float_tensor_float16=torch.tensor([1,2,3,4],dtype=torch.float32)

float_tensor_float16=torch.tensor([1,2,3,4]
                                  ,dtype=torch.float16 ## data type of tensor
                                  ,device="cuda" ## what device your tensor is on
                                  ,requires_grad=False ## whether or not to track gradients with tensor operation
                                  )
float_tensor_float16=torch.tensor([1,2,3,4],dtype=torch.float32,device="cpu") ## for using cpu

#find out details about some tensor
print(m.dtype)

"""Manipulating Tensor
addition
subtraction
miltiplication
division
matrix multiplication
"""

test_tensor=torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
test_tensor=test_tensor*10
print(test_tensor)

## pytorch built in function
# for matrix multiplication
# inner dimension must be same
# resulting matrix has dimension of outer dimension

torch.mul(test_tensor,10)

torch.matmul(test_tensor,test_tensor)# Matrix multiplication

#to munipulate the shape of out tensor we can use Transpose
# Transpose change axex of given tensor
print(test_tensor.T)

p=torch.arange(0,100,5)

#Find MIN
print("MIN is _____")
print(p.min)
print(torch.min(p))

#Find MAX
print("MAX is _____")
print(p.max)
print(torch.max(p))

#Find MEAN
# requires float 32 =========
print("MEAM is _____")
p.type(torch.float32).mean, torch.mean(p.type(torch.float32))

#find index of min max
# min- p.argmin()
# max- p.argmax()

"""## Reshaping, Stacking, Squezing and Unsqueezing tensor
* Reshaping- reshapes an input tensor to a defined shape
* View- Return a view of an input tensor of certain shape but keep the same memory as the orignal tensor
* Stacking- combine multiple tensor on top of each other (vstack) or side by side (hstack)
* Squeeze - removes all '1' dimensions from a tensor
* Unsqueeze add a 1 dimensions to a target tensor
* Peremute- return a view of the input with dimension permuted(swapped) in a certain way

"""

p,p.shape

a=p.reshape(2,10) # change shape to desired shape number of element must be same 20=2*10=4*5
a

b=p.reshape(4,5)
b

c=p.view(5,4) # only changes view of it changing c will change p as p and c have same address
c,p

# Stacking tensors
p_stacked=torch.stack([p,p,p,p,p,p],dim=0)
p_stacked

"""##Squeeze
remove all single dimension from the tensor
a=b.squeeze()
## Unsqueeze
adds a single dimension to a target tensor at a specific dimension

"""

a=torch.tensor([[1],[2],[3],[4]])
print(a)
print(a.shape)

b=a.squeeze()
print(b)
print(b.shape)

c=b.unsqueeze(dim=0)
print(c)
print(c.shape)

# permute
# return a view of the original tensor with its dimension permuted (changed)

a9=torch.randn(2,3,4)
print(a9.size())
a10=torch.permute(a9,(2,0,1))
print(a10.size())

"""# Pytorch AND Numpy
##Data in Numpy to Tensor
  "torch.from_numpy(ndarray)"
##Data in Tensor to Numpy
  "torch.tensor.numpy()"

"""

import torch
import numpy as np

array=np.arange(1.0,8.0)
print(array)
tensor=torch.from_numpy(array) ### numpy to tensor
print(tensor)
arr=torch.Tensor.numpy(tensor)  ### Tensor to Numpy
print(arr)

"""#Reproducablity
making random values less random


## to reduce this we use random seed
"""

import torch
random_a=torch.rand(3,4)
random_b=torch.rand(3,4)
print(random_a)
print(random_b)
print(random_a==random_b)

"""#Random but reproducible seed"""

import torch
rand_seed=100
torch.manual_seed(rand_seed) ### only work for the next random method this wont work for rand_b if you want to use it for b then type it again before b
rand_a=torch.rand(3,4)
torch.manual_seed(rand_seed)
rand_b=torch.rand(3,4)
print(rand_a)
print(rand_b)
print(rand_a==rand_b)