import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from model import predict_probability 
from loss import log_loss
from train import train
from decision import apply_threshold
from evaluation_metrics import confusion_matrix,accuracy,precision,recall

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
#print(w)
#print(X)

y_pred_bt=[predict_probability(xi,w,b) for xi in X]

plt.figure(figsize=(6,4))
plt.title("Model's prediction - before training")
plt.plot(y_pred_bt)
plt.grid(True)
#plt.savefig("model_prediction_before_training.png")
plt.show()

w,b,losses=train(X,y)

y_pred_at=[predict_probability(xi,w,b) for xi in X]

plt.figure(figsize=(6,4))
plt.title("Model's prediction - after training")
plt.plot(y_pred_at)
plt.grid(True)
#plt.savefig("model_prediction_after_training.png")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(list(range(100)),losses)
plt.title("Loss vs iterations")
plt.grid(True)
#plt.savefig("loss_vs_decreases.png")
plt.show()

thresholds = [0.2,0.5,0.9]

all_predictions={}

for t in thresholds:
    predictions=[apply_threshold(p,t) for p in y_pred_at]
    all_predictions[t]=predictions
    print(t,predictions)

for t,predictions in all_predictions.items():
    TP,FP,TN,FN=confusion_matrix(y,predictions)

    acc=accuracy(TP,FP,TN,FN)
    prec=precision(TP,FP)
    rec=recall(TP,FN)

    print(f"Threshold = {t}")
    print(f" TP={TP} \n FP={FP} \n FP={FP}\n TN={TN}")
    print(f" Accuracy={acc:.2f}\n Precision={prec:.2f} \n Recall={rec:.2f}")