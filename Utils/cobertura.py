import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import argparse
from sklearn.metrics import roc_curve as ROC
from sklearn.metrics import auc,brier_score_loss
import numpy as np

def correct_class(y_pred,y_true):
    '''Returns a bool tensor indicating if each prediction is correct'''
    with torch.no_grad():
        #y_pred = torch.argmax(y_pred, -1) # não entendi para que serve
        correct = (y_pred==y_true)
    return correct

def correct_total(y_pred,y_true):
    '''Returns the number of correct predictions in a batch'''
    with torch.no_grad():
        correct = correct_class(y_pred,y_true)
        correct_total = torch.sum(correct).item()
    return correct_total


def get_n_biggest(vec,n):
    '''Returns the indexes of the N biggest values in vec'''
    if 0<n<1:
        n = int(n*vec.size(0))
    unc = torch.argsort(vec, descending = True)
    return unc[0:n]

def masked_coverage(y_pred,y_true, uncertainty, coverage):
    
    #dk_mask = unc_utils.dontknow_mask(uncertainty, coverage)
    #y_pred, y_true = torch.masked_select(y_pred,1-dk_mask),torch.masked_select(y_true,1-dk_mask)#apply_mask(y_pred,y_true,1-dk_mask)
    N = round((coverage)*uncertainty.shape[0])
    id = get_n_biggest(uncertainty,N)
    y_pred = y_pred[id]
    y_true = y_true[id]
    
    return y_pred,y_true

def acc_coverage(y_pred,y_true, uncertainty, coverage):
    '''Returns the total accuracy of model in some dataset excluding the 1-c most uncertain samples'''
    y_pred,y_true = masked_coverage(y_pred,y_true, uncertainty, coverage)
    acc = correct_total(y_pred,y_true)/y_true.size(0)
    return acc

def error_coverage(y_pred,y_true, uncertainty, coverage):
    '''Returns the 0-1 loss of model in some dataset excluding the 1-c most uncertain samples'''
    return 1-acc_coverage(y_pred,y_true, uncertainty, coverage)

def RC_curve(y_pred,y_true,uncertainty, risk = error_coverage, c_list = np.arange(0.05,1.05,0.05)):
    ''' Returns an array with the accuracy of the model in the data dataset
     excluding the most uncertain (total number set by the coverage) samples.
     Each item in the output array is the accuracy when the coverage is given by same item in c_list'''

    risk_list = np.array([])
    with torch.no_grad():
        for c in c_list:
            r = risk(y_pred,y_true, uncertainty, c)
            risk_list = np.append(risk_list,r)
    return risk_list

def entropy(y):
        '''Returns the entropy of a probabilities tensor.'''
  
        entropy = torch.special.entr(y) #entropy element wise
        entropy = torch.sum(entropy,-1)

        return entropy

def mutual_info(pred_array):
    '''Returns de Mutual Information (Gal, 2016) of a ensemble
pred_array deve ser um tensor contendo os outputs (probabilidades) das T redes constituintes do Ensemble; Ou seja, deve ter shape (T,N,K), 
onde N é o número de pontos e K o número de classes do dataset. No caso de um Ensemble de 10 redes em todo o conjunto de teste do Cifar10, 
o shape será (10,10000,10)
'''
    ent = -entropy(torch.mean(pred_array, axis=0))
    MI = ent - torch.mean(entropy(pred_array), axis=0) 
    return MI

def ROC_curve(output,y_true, uncertainty, return_threholds = False):
    if callable(uncertainty):
        uncertainty = uncertainty(output)
    y_true = np.logical_not(correct_class(output,y_true).cpu().numpy())
    fpr, tpr, thresholds = ROC(y_true,uncertainty.cpu().numpy())
    if return_threholds:
        return fpr,tpr,thresholds
    else:
        return fpr,tpr

def AURC(y_pred,y_true,uncertainty, risk = error_coverage, c_list = np.arange(0.05,1.05,0.05)):
    risk_list = RC_curve(y_pred,y_true,uncertainty, risk, c_list)
    return np.trapz(risk_list,x = c_list, axis = -1)

def AUROC(output,y_true,uncertainty):
    fpr,tpr = ROC_curve(output,y_true,uncertainty)
    return auc(fpr, tpr)