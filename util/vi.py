# Variation of information (VI)
#
# Meila, M. (2007). Comparing clusterings-an information
#   based distance. Journal of Multivariate Analysis, 98,
#   873-895. doi:10.1016/j.jmva.2006.11.013
#
# https://en.wikipedia.org/wiki/Variation_of_information

from math import log
import numpy as np

def variation_of_information_ori(X, Y):#X的总元素个数和Y的总元素个数相同                
  n = float(sum([len(x) for x in X]))#总元素个数
  sigma = 0.0
  for x in X:#X中每个元素都会和Y中的每个元素计算r
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n#交集元素个数 如果只有0 1标签就会只剩下{0.0, 1.0}
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma)

def variation_of_information(X, Y):#X的总元素个数和Y的总元素个数相同 
  assert len(X) == len(Y)
  assert X.shape == Y.shape
  n = float(len(X))               
  # n = float(sum([len(x) for x in X]))#n是总元素个数 X和Y是两种关于整个集合A的划分方式，X中的每个x可以不同长度 正常来说集合是不能有相同元素的 这里把不同像素出来的0看做不同的元素 因此不能用set 而要用异或 X中的每个x都要和Y中的每一个y计算

  r = np.sum([bool(x1) ^ bool(y1) for x1,y1 in zip(X, Y)]) / n
  p = len(X) / n
  q = len(Y) / n
  sigma = r * (log(r / p, 2) + log(r / q, 2))

  return abs(sigma)

if __name__ == "__main__":
  # Identical partitions
  X1 = [ [1,2,3,4,5], [6,7,8,9,10] ]
  Y1 = [ [1,2,3,4,5], [6,7,8,9,10] ]
  print(variation_of_information(X1, Y1))
  # VI = 0

  # Identical partitions
  X1 = [ [1],[6]]
  Y1 = [ [1],[5]]
  print(variation_of_information(X1, Y1))

  # Identical partitions
  X1 = [ [1,2,3,4,6]]
  Y1 = [ [1,2,3,4,5]]
  print(variation_of_information(X1, Y1))

  # Similar partitions
  X2 = [ [1,2,3,4], [5,6,7,8,9,10] ]
  Y2 = [ [1,2,3,4,5,6], [7,8,9,10] ]
  print(variation_of_information(X2, Y2))
  # VI = 1.102

  # Dissimilar partitions
  X3 = [ [1,2], [3,4,5], [6,7,8], [9,10] ]
  Y3 = [ [10,2,3], [4,5,6,7], [8,9,1] ]
  print(variation_of_information(X3, Y3))
  # VI = 2.302

  # Totally different partitions
  X4 = [ [1,2,3,4,5,6,7,8,9,10] ]
  Y4 = [ [1], [2], [3], [4], [5], [6], [7], [8], [9], [10] ]
  print(variation_of_information(X4, Y4))
  # VI = 3.322 (maximum VI is log(N) = log(10) = 3.322)