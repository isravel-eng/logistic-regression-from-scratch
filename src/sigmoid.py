import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(z):
    return 1/(1+np.exp(-z))

z=np.linspace(-10,10,200)
y=sigmoid(z)

plt.figure(figsize=(10,6))
plt.plot(z,y)
plt.xlabel("Raw scores(z)")
plt.ylabel("Probability")
plt.title("Sigmoid function mapping the raw scores to probabilities")
plt.legend()
plt.grid(True)
plt.show()