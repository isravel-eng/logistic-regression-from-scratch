import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict_probability(x, weights, bias):
    score = sum(weights[i] * x[i] for i in range(len(x))) + bias
    return sigmoid(score)
