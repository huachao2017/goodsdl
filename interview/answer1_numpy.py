"""
用python编程：有一个4*4的二维矩阵，请按如下图顺序输出里面的值。
"""

import numpy as np

input = np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
input = input.reshape(4,4)

def output_edge(input, output,direction = 0):
  if(input.size == 0):
    return
  if direction == 0:
    output.append(input[0,:].tolist())
    output_edge(input[1:,:],output,1)
  elif direction == 1:
    output.append(input[:,input.shape[1]-1].tolist())
    output_edge(input[:,:input.shape[1]-1],output,2)
  elif direction == 2:
    t = input[input.shape[0]-1,:]
    t = t[::-1]
    output.append(t.tolist())
    output_edge(input[:input.shape[0]-1,:],output,3)
  elif direction == 3:
    t = input[:,0]
    t = t[::-1]
    output.append(t.tolist())
    output_edge(input[:,1:],output,0)


output = []
output_edge(input,output)
print(output)

