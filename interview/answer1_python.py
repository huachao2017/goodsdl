"""
用python编程：有一个4*4的二维矩阵，请按如下图顺序输出里面的值。
"""

import numpy as np

input = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

def output_edge(input, output,direction = 0):
  if(len(input) == 0 or len(input[0]) == 0):
    return
  if direction == 0:
    output.append(input[0])
    input.pop(0)
    output_edge(input,output,1)
  elif direction == 1:
    row = len(input)
    column = len(input[0])
    output.append([input[i][column-1] for i in range(row)])
    for i in range(row):
      input[i].pop(column-1)
    output_edge(input,output,2)
  elif direction == 2:
    row = len(input)
    column = len(input[0])
    output.append([input[row-1][column-i-1] for i in range(column)])
    input.pop(row-1)
    output_edge(input,output,3)
  elif direction == 3:
    row = len(input)
    column = len(input[0])
    output.append([input[row-i-1][0] for i in range(row)])
    for i in range(row):
      input[i].pop(0)
    output_edge(input,output,0)


output = []
output_edge(input,output)
print(output)

