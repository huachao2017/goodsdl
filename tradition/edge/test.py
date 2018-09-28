import cv2
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def pooling(feature_map, size=2, stride=2):
  # Preparing the output of the pooling operation.
  pool_out = np.zeros((np.uint16((feature_map.shape[0] - size + 1) / stride),
                          np.uint16((feature_map.shape[1] - size + 1) / stride),
                          feature_map.shape[-1]))
  for map_num in range(feature_map.shape[-1]):
    r2 = 0
    for r in np.arange(0, feature_map.shape[0] - size - 1, stride):
      c2 = 0
      for c in np.arange(0, feature_map.shape[1] - size - 1, stride):
        pool_out[r2, c2, map_num] = np.max([feature_map[r:r + size, c:c + size]])
        c2 = c2 + 1
      r2 = r2 + 1
  return pool_out

base_dir, _ = os.path.split(os.path.realpath(__file__))
image_dir = os.path.join(base_dir, 'images')
depth_path = os.path.join(image_dir, "04_d.png")

img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)#, flags=cv2.COLOR_BGR2GRAY)
# print(img.shape)
# print(img)
depth = img[:,:,0]+img[:,:,1]*256+img[:,:,2]*256*256
depth = np.expand_dims(depth,2)
depth = pooling(depth,2,2)
depth = pooling(depth,2,2)
depth = pooling(depth,2,2)
depth = depth[:,:,0]
print(depth.shape)
# print(a.shape)
# print(a)
# b = np.where(a < 1240, a, np.zeros(a.shape))
# b = np.where(b > 4096, np.zeros(a.shape), b)
# print(b)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.arange(0,depth.shape[0])
X = X.repeat(depth.shape[1])
print(X)


Y = np.arange(0,depth.shape[1])
Y = np.expand_dims(Y, 0)
Y = Y.repeat(depth.shape[0],axis=0)
Y = Y.reshape([depth.shape[0]*depth.shape[1]])
print(Y)

z_0 = 1235

depth = np.where(depth > z_0, np.full(depth.shape,z_0), depth)
depth = np.where(depth < 10, np.full(depth.shape,z_0), depth)
depth = z_0 - depth
depth = np.where(depth < 20, np.zeros(depth.shape), depth)
Z = depth

Z = Z.reshape([depth.shape[0]*depth.shape[1]])
ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
plt.show()
# cv2.imshow('test',img)  #显示图片
# if cv2.waitKey(0) == 27:
#   cv2.destroyAllWindows()
