#로지스틱 회귀 분류

import numpy as np
import matplotlib.pyplot as plt

#sigmoid 함수

def sigmoid(x, a=1, b=0):
    return (1. / (1+np.exp(-a*(x-b)))) 

xs = np.linspace(-5,5,1001)
ys = sigmoid(xs)

plt.plot (xs,ys,label='sigmoid')
plt.plot(xs, ys*(1-ys), label='derivative')
plt.title('Sigmoid function')
plt.yticks([0,0.5,1])
plt.grid()
plt.legend()
plt.show()

y3 = sigmoid(xs, a=3)
y_half = sigmoid(xs, a=0.5)
plt.plot (xs,ys,label='sigmoid')
plt.plot (xs,y3,label='sigmoid *3')
plt.plot (xs,y_half,label='sigmoid *0.5')

plt.title('Sigmoid function')
plt.yticks([0,0.5,1])
plt.grid()
plt.legend()
plt.show()