import torch

eps = 1e-6

def accuracy(TN, FP, FN, TP):
    return (TP + TN) / (TN + FP + FN + TP + eps)

def precision(TN, FP, FN, TP):
    return TP / (FP + TP + eps)

def recall(TN, FP, FN, TP):
    return TP / (FN + TP + eps)

def specificity(TN, FP, FN, TP):
    return TN / (FP + TN + eps)

def f1_score_fun(TN, FP, FN, TP):
    p = precision(TN, FP, FN, TP)
    r = recall(TN, FP, FN, TP)
    return 2 * p * r / (p + r + eps)

def tss(TN, FP, FN, TP):
    return recall(TN, FP, FN, TP) + specificity(TN, FP, FN, TP) - 1

def gmean(TN, FP, FN, TP):
    sens = TP / (TP + FN + eps) 
    spec = TN / (TN + FP + eps)
    return torch.sqrt(sens * spec)