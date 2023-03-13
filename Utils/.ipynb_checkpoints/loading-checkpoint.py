import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from Utils.redes import *
from Utils.ensemble import DeepEnsemble

def load_student(students, data_set,n_redes=15, temp1=2, temp2=3):
    # CARREGA TODOS OS STUDENTS
    for dirname, _, filenames in os.walk('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble/{} Redes/'.format(n_redes)):
        for filename in filenames:
            if filename[0:8] == "student_":
                students[0].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 1/Temp {}/{}'.format(data_set,n_redes,temp1,filename)))
                students[0].eval()
            if filename[0:8] == "student2":
                students[1].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 1/Temp {}/{}'.format(data_set,n_redes,temp1,filename)))
                students[1].eval()
            if filename[0:8] == "student3":          
                students[2].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 2/Temp {}/{}'.format(data_set,n_redes,temp2,filename)))
                students[2].eval()
            if filename[0:8] == "student4":
                students[3].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 2/Temp {}/{}'.format(data_set,n_redes,temp2,filename)))
                students[3].eval()
    return students

def load_cifar10(n_nets):            
    # CARREGA AS REDES DO CIFAR 10 DE UMA MANEIRA QUE POSSA SER USADA A CLASE DEEPENSEMBLE
    nets = list()
    for i in range(n_nets):
        if i == 0:
            i = ""     
        net = torch.load("/home/luigi-doria/IC/Data_sets/Cifar10/net{}.pth".format(i)).module.to("cuda")
        net_conv = ResNet18().to("cuda")
        net_conv.load_state_dict(net.state_dict())
        net_conv.eval()
        nets.append(net_conv)     
    return nets

def load_cifar100(n_redes):
    # CARREGA AS REDES DO CIFAR 100 DE UMA MANEIRA QUE POSSA SER USADA A CLASE DEEPENSEMBLE
    nets = list()
    for dirname, _, filenames in os.walk('/home/luigi-doria/IC/Data_sets/Cifar100'):
        for i, filename in enumerate(filenames):
            if i == n_redes:
                break
            net = torch.load("/home/luigi-doria/IC/Data_sets/Cifar100/{}".format(filename)).module.to("cuda")
            net_conv = ResNet18_100().to("cuda")
            net_conv.load_state_dict(net.state_dict())
            net_conv.eval()
            nets.append(net_conv)
    return nets

def load_resnet18(n_redes, data_set):
    # APENAS CARREGA AS REDES DO CIFAR 10 OU CIFAR 100
    nets = list()
    if data_set == "Cifar10":
        for i in range(n_redes):
            if i == 0:
                i = ""
            nets.append(torch.load('/home/luigi-doria/IC/Data_sets/Cifar10/net{}.pth'.format(i)))
    elif data_set == "Cifar100": 
        for dirname, _, filenames in os.walk('/home/luigi-doria/IC/Data_sets/Cifar100'):
            for i, filename in enumerate(filenames):
                if i == n_redes:
                    break
                nets.append(torch.load("/home/luigi-doria/IC/Data_sets/Cifar100/{}".format(filename)).module.to("cuda"))
    return nets

def load_student_resnet(students, data_set,n_redes=15, temp1=2, temp2=3):
    # FAZ O LOADING APENAS DOS STUDENTS RESNET
    for dirname, _, filenames in os.walk('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/'.format(data_set,n_redes)):
        for filename in filenames:
            if filename[0:8] == "student_" and dirname == '/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 1/Temp {}'.format(data_set,n_redes,temp1):
                students[0].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 1/Temp {}/{}'.format(data_set,n_redes,temp1,filename)))
                students[0].eval()
            if filename[0:8] == "student3"and dirname == '/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 2/Temp {}'.format(data_set,n_redes,temp2):
                students[1].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Ensemble {}/{} Redes/Metodo 2/Temp {}/{}'.format(data_set,n_redes,temp2,filename)))
                students[1].eval()
    return students

def load_ensamble(n_nets, data_set):
    # 
    if data_set == "Cifar10":
        nets = load_resnet18(n_nets)
    elif data_set == "Cifar100":
        nets = load_cifar100(n_nets)
    ensamble = DeepEnsemble(nets, apply_softmax=True)
    return ensamble

def load_teste_temp(students, data_set, temp):
    # FAZ O LOADING APENAS DOS STUDENTS RESNET
    for dirname, _, filenames in os.walk('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Teste de Temperatura/{}/Temp {}/'.format(data_set,temp)):
        for filename in filenames:
            if filename[0:8] == "student_":
                students[0].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Teste de Temperatura/{}/Temp {}/{}'.format(data_set,temp,filename)))
                students[0].eval()
            if filename[0:8] == "student3":          
                students[1].load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Knowledge_distillation/Teste de Temperatura/{}/Temp {}/{}'.format(data_set,temp,filename)))
                students[1].eval()
    return students

def load_hydra(hydra, data_set, n_cabecas):
    # FAZ O LOADING APENAS DOS STUDENTS RESNET
    n_cabecas = str(n_cabecas)
    if len(n_cabecas) == 1:
        n_cabecas = "{}_".format(n_cabecas)

    for dirname, _, filenames in os.walk('/home/luigi-doria/IC/Data_sets/Hydra {}/'.format(data_set)):
        for filename in filenames:
            if filename[6:8] == n_cabecas:
                hydra.load_state_dict(torch.load('/home/luigi-doria/IC/Data_sets/Hydra {}/{}'.format(data_set,filename)))
    return hydra