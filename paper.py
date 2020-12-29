import matplotlib.pyplot as plt
plt.style.use('science')

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
with plt.style.context(['science', 'ieee']):
    plt.figure()
    plt.plot(x, x)
    plt.show()
