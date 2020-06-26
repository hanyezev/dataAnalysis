import numpy as np
import matplotlib.pyplot as plt

arr_height = np.random.normal(175, 10, size=1000000)

sample_height = np.random.choice(arr_height, size=1, replace=True)

average = []

n = 10000

for i in range(n):
    sample = np.random.choice(arr_height, size=1, replace=True)
    sample_height = np.append(sample_height, sample)
    average.append(np.average(sample_height))

plt.figure(figsize=(8, 6))
plt.plot(np.arange(n), average, alpha=0.6, color='blue')
plt.plot(np.arange(n), [175 for i in range(n)], alpha=0.6, color='red', linestyle='--')
plt.xlabel('Sample Rounds', fontsize=10)
plt.show()