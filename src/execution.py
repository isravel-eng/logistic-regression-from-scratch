import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from model import predict_probability 
from loss import log_loss
from train import train
from decision import apply_threshold

X = [
    [0.2, 0.1],
    [0.4, 0.3],
    [0.6, 0.5],
    [1.2, 1.1],
    [1.4, 1.3],
    [1.6, 1.5],
]

y = [0, 0, 0, 1, 1, 1]

w=[0.0]*len(X[0])
b=0.0
print(w)
print(X)
y_pred_bt=[predict_probability(w,b,xi) for xi in X]

plt.figure(figsize=(6,4))
plt.title("Model's prediction - before training")
plt.plot(y_pred_bt)
plt.grid(True)
plt.savefig("model_prediction_before_training.png")

w,b,losses=train(X,y)

y_pred_at=[predict_probability(w,b,xi) for xi in X]

plt.figure(figsize=(6,4))
plt.title("Model's prediction - after training")
plt.plot(y_pred_at)
plt.grid(True)
plt.savefig("model_prediction_after_training.png")


plt.figure(figsize=(10,6))
plt.plot(list(range(100)),losses)
plt.title("Loss vs iterations")
plt.grid(True)
plt.savefig("loss_vs_decreases.png")