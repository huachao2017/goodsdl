"""
用python编程：有一个4*4的二维矩阵，请按如下图顺序输出里面的值。
"""

import tensorflow as tf
from tensorflow.python.framework import ops


"""
通过tensor来实现旋转遍历矩阵的算法
tensor进入递归函数没有问题，但是逻辑分支不能基于tensor在运行时的值，关于type，shape的值是在运行前就决定的。
tf.split: 张量分解
tf.concat: 张量合并
tf.reverse: 张量反序
tf.transpose: 张量转置
"""
def get_path(in_tensor, out_tensor, is_first=True):
  if not is_first:
    if in_tensor.get_shape().as_list()[0] > 1:
      in_tensor = tf.reverse(in_tensor, [1])
      in_tensor = tf.transpose(in_tensor)
      cur_value, cur_matrix = tf.split(in_tensor, [1, in_tensor.get_shape().as_list()[0] - 1], 0)
      out_tensor = tf.concat([tf.squeeze(out_tensor), tf.squeeze(cur_value)], 0)
      return get_path(cur_matrix, out_tensor, is_first=False)
    else:
      in_tensor = tf.reverse(in_tensor, [1])
      out_tensor = tf.concat([tf.squeeze(out_tensor), tf.squeeze(in_tensor)], 0)
      return out_tensor
  else:
    cur_value, cur_matrix = tf.split(in_tensor, [1, in_tensor.get_shape().as_list()[0] - 1], 0)
    out_tensor = tf.squeeze(cur_value)
    return get_path(cur_matrix, out_tensor, is_first=False)


if __name__ == '__main__':
  ops.reset_default_graph()
  sess = tf.Session()
  a = tf.range(1, 17, 1)
  a = tf.reshape(a, [4, 4])
  b = get_path(a, None)
  c = sess.run(b)
  print(c)