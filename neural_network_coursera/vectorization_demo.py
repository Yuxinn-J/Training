import numpy as np
import time

a = np.array([1, 2, 3, 4])
print(a)

# create a million dimensional array with random values
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# vectorised version
# measure the current time
tic = time.time()
d = np.dot(a, b)
toc = time.time()
print(d)
print("Vectorized version: " + str(1000*(toc-tic)) + " ms")


# non-vectorised version
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("For loop: " + str(1000*(toc-tic)) + " ms")

