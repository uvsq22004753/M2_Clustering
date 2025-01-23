import numpy as np
import time

size = 10**7
np_array = np.arange(size)
py_list = list(range(size))

start_time = time.time()
for _ in range(1000):
    index = np.random.randint(size)
    value = np_array[index]
np_time = time.time() - start_time

start_time = time.time()
for _ in range(1000):
    index = np.random.randint(size)
    value = py_list[index]
py_time = time.time() - start_time

print(f"NumPy array random access time: {np_time:.6f} seconds")
print(f"Python list random access time: {py_time:.6f} seconds")