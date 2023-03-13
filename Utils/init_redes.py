import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from Utils.redes import *

if torch.cuda.is_available():
    device = 'cuda'  
else:
    device = 'cpu'

def init_rede(nome, num_cabeca=0):

    if nome == "Resnet":
        rede = ResNet18()
    elif nome == "VGG11":
        rede = VGG('VGG11')
    elif nome == "Resnet100":
        rede = ResNet18_100()
    elif nome == "Resnet_hydra":
        rede = ResNet18_hydra(num_cabeca)
    elif nome == "Resnet100_hydra":
        rede = ResNet18_hydra_100(num_cabeca)
        
    rede = rede.to(device)
    if device == 'cuda':
        rede = torch.nn.DataParallel(rede)
        cudnn.benchmark = True
    
    return rede

def init_rede_all(nome, num_cabeca=0):

    if nome == "Resnet":
        rede = ResNet18()
    elif nome == "VGG11":
        rede = VGG('VGG11')
    elif nome == "Resnet100":
        rede = ResNet18_100()
    elif nome == "Resnet_hydra":
        rede = ResNet18_hydra(num_cabeca)
    elif nome == "Resnet100_hydra":
        rede = ResNet18_hydra_100(num_cabeca)
        
    rede = rede.to(device)
    if device == 'cuda':
        rede = torch.nn.DataParallel(rede)
        cudnn.benchmark = True
        
    loss_criterion = nn.CrossEntropyLoss()
    optimizer_rede = optim.SGD(rede.parameters(), lr=0.2,
                          momentum=0.9, weight_decay=5e-4)
    scheduler_rede = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_rede, T_max=200)
    
    return rede, loss_criterion, optimizer_rede, scheduler_rede