import numpy as np
import time

A = np.random.rand(10, 10)
B = np.random.rand(10, 10)

start = time.time()
C = A @ B
end = time.time()
t1 = end - start
print(C)
print(t1)

start = time.time()
D = np.dot(A, B)
end = time.time()
t2 = end - start
print(D)
print(t2)

print(f"Quotient t2/t1: {t2 / t1}")
