from train import train 
from model import predict_probability
from evaluation_metrics import evaluate_model

X = [
    [0.2, 0.1],
    [0.4, 0.3],
    [0.6, 0.5],
    [1.2, 1.1],
    [1.4, 1.3],
    [1.6, 1.5],
]

y = [0, 0, 0, 1, 1, 1]

weights,bias,losses=train(X,y)
y_pred=[predict_probability(x,weights,bias) for x in X]

thresholds=[0.2,0.5,0.9]
evaluate_model(y,y_pred,thresholds)