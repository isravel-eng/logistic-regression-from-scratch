import math

def log_loss(y_true,y_pred):
    eps=1e-15
    y_pred= min(max(y_pred,eps),1-eps)
    return -(y_true*math.log(y_pred)+(1-y_true)*(math.log(1-y_pred)))