import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])

# to sum vertically axis=0
# to sum horizontally axis=1
cal = A.sum(axis=0)
print(cal)

# A: 3*4 matrix
# cal 1*4 matrix
percentage = 100 * A / cal.reshape(1, 4)
print(percentage)

