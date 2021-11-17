from build.python import data, gpu
from build.python.tier1 import *


import numpy as np

print("data wrapper testing: minimal interaction with data type buffer and image")
print("-------------------------------")
print("buffer test")
buff = data.buffer()
print(buff.info())
print(buff.shape())
print(buff.dtype())
print("-------------------------------")
print("image test")
img = data.image()
print(img.info())
print(img.shape())
print(img.dtype())



print("gpu wrapper testing: select gpu, and push/pull/create datatype")
print("-------------------------------")
print("select gpu")
dev2 = gpu.gpu("NVIDIA")
print(dev2.info())

print("-------------------------------")
print("create buffer")
buff2 = dev2.create_buffer(np.asarray([2,2]))
print(buff2.info())

print("-------------------------------")
print("push buffer")
arr = np.ones([2,2]) * 100.0
buff3 = dev2.push_buffer(arr)
print(buff3.info())

print("-------------------------------")
print("pull buffer")
res = dev2.pull_buffer(buff3)
print(res.shape, " vs ", arr.shape)
print(np.sum(res.flatten() - arr.flatten()))

print("-------------------------------")
print("create buffer")
buff2 = dev2.create_buffer(np.asarray([10,5,2]))
print(buff2.info())

print("-------------------------------")
print("push buffer")
arr = np.ones([10,5,2]) * 100
buff3 = dev2.push_buffer(arr)
print(buff3.info())

print("-------------------------------")
print("pull buffer")
res = dev2.pull_buffer(buff3)
print(res.shape, " vs ", arr.shape)
print(np.sum(res.flatten() - arr.flatten()))


print("-------------------------------")
print("create image")
img = dev2.create_image(np.asarray([10,5,2]))
print(img.info())

print("-------------------------------")
print("push image")
arr = np.ones([10,5,2]) * 100
img2 = dev2.push_image(arr)
print(img2.info())

print("-------------------------------")
print("pull image")
res = dev2.pull_image(img2)
print(res.shape, " vs ", arr.shape)
print(np.sum(res.flatten() - arr.flatten()))



print("kernel wrapper testing: run a kernel class (low-level)")
print("-------------------------------")
arr = np.ones([5,2,3]) * 100.0
val = np.ones([5,2,3]) * 100.0 + 100
input = dev2.push_buffer(arr)
output = dev2.create_buffer(arr.shape)
add_image_and_scalar(input, output, 100, dev2)
res = dev2.pull_buffer(output)
print(res.shape, " vs ", val.shape)
print(np.sum(res.flatten() - val.flatten()))
