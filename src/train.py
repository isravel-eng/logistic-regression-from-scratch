from model import predict_probability
from loss import log_loss

def train(X,y,lr=0.1,epochs=100):
    n_features= len(X[0])
    weights=[0.0]*n_features
    bias = 0.0
    losses = []

    for epoch in range(epochs):
        total_loss=0
        
        for xi,yi in zip(X,y):
            prob = predict_probability(xi,weights,bias)
            loss = log_loss(yi,prob)
            total_loss+=loss

            # gradients
            error = prob - yi
            for j in range(n_features):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error

        losses.append(total_loss/len(X))
    return weights,bias,losses