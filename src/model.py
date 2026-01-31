import math 

def sigmoid(z):
    return 1/(1+math.exp(-z))

def predict_probability(weights,bias,x):
    score= sum(weights[i]*x[i] for i in range(len(x))) + bias
    return sigmoid(score)
