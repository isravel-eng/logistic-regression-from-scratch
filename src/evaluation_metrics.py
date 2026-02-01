from decision import apply_threshold

def confusion_matrix(y_true, y_pred):
    TP=FP=TN=FN=0

    for y,p in zip(y_true,y_pred):
        if y==1 and p==1:
            TP+=1
        elif y==1 and p==0:
            FN+=1
        elif y==0 and p==1:
            FP+=1
        elif y==0 and p==0:
            TN+=1
    return TP,FP,TN,FN

def accuracy(TP,FP,TN,FN):
    return (TP + TN)/(TP+FP+TN+FN)

def precision(TP,FP):
    return TP/(TP+FP) if (TP+FP) > 0 else 0

def recall(TP,FN):
    return TP/(TP+FN) if (TP+FN) > 0 else 0

def evaluate_model(y_true,y_pred,thresholds):
    
    for t in thresholds:
        predictions=[apply_threshold(p,t) for p in y_pred]
        TP,FP,TN,FN=confusion_matrix(y_true,predictions)

        acc=accuracy(TP,FP,TN,FN)
        prec=precision(TP,FP)
        rec=recall(TP,FN)

        print(f"Threshold = {t}")
        print(f" TP={TP},  FP={FP},  FP={FP}, TN={TN}")
        print(f" Accuracy={acc:.2f}\n Precision={prec:.2f} \n Recall={rec:.2f}\n --------------------")
