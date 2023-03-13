import matplotlib
import matplotlib.pyplot as plt
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
from Utils.graficos import *
from Utils.cobertura import *

def calculate_correct(predicted, labels):
    correct_aux = (predicted == labels.to("cuda")).sum().item()
    return round(1-correct_aux/10000,4)

def calculate_predicted(redes):
    uncs_max_list = list()
    predicted_list = list()
    for rede in redes:
        uncs_max, predicted_aux = torch.max(rede.data, 1)
        uncs_max_list.append(uncs_max)
        predicted_list.append(predicted_aux)
    predict_cat = torch.stack(tuple(predicted_list),dim=0) # Junta as respostas
    uncs_max_cat = torch.stack(tuple(uncs_max_list),dim=0) # Junta as respostas
    return uncs_max_cat, predict_cat

def calculate_var(rede, redes=1):
    uncs_var_list = list()
    for rede in redes:
        uncs_var = torch.var(rede,redes)
        uncs_var_list.append(uncs_var)
    uncs_var_cat = torch.stack(tuple(uncs_var_list),dim=0) # Junta as respostas
    return uncs_var_cat

def calculate_entr(redes):
    uncs_sum_entr = list()
    for rede in redes:
        uncs_entr = torch.special.entr(rede)
        uncs_sum_entr = -torch.sum(uncs_entr, dim=-1)
        uncs_sum_entr.append(uncs_sum)
    uncs_entr_cat = torch.stack(tuple(uncs_entr_list),dim=0) # Junta as respostas
    return uncs_entr_cat

def calculate_correct_list(predict_cat, labels_cat):
    correct_list = list()
    for predict in predict_cat:
        correct_list.append(calculate_correct(predict, labels_cat))
    return correct_list

def calculate_covarege(predict_cat, uncs, labels_cat):
    covarege_list = list()
    for i in range(len(predict_cat)):
        covarege_list.append(RC_curve(predict_cat[i],labels_cat.to("cuda"),uncs[i]))  
    if len(predict_cat) == 1:
        return covarege_list[0]
    else:
        return covarege_list

def ROC_curve_list(predict_cat, uncs, labels_cat):
    roc_curve1 = list()
    roc_curve2 = list()
    for i in range(len(predict_cat)):
        TPR, FPR = ROC_curve(predict_cat[i],labels_cat.to("cuda"),-uncs[i])
        roc_curve1.append(TPR)
        roc_curve2.append(FPR)
    if len(predict_cat) == 1:
        return roc_curve1[0], roc_curve2[0]
    else:
        return roc_curve1, roc_curve2

def AUROC_curve_list(predict_cat, uncs, labels_cat):
    auroc_max = list()
    for i in range(len(predict_cat)):
        auroc_max.append(AUROC(predict_cat[i],labels_cat.to("cuda"),-uncs[i]))
    if len(predict_cat) == 1:
        return auroc_max[0]
    else:
        return auroc_max

def AURC_curve_list(predict_cat, uncs, labels_cat):
    aurc = list()
    for i in range(len(predict_cat)):
        aurc.append(AURC(predict_cat[i],labels_cat.to("cuda"),-uncs[i]))
    if len(predict_cat) == 1:
        return aurc[0]
    else:
        return aurc

