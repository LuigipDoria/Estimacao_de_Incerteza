import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class Ensemble(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    
    def forward(self,x):
        self.ensemble = self.get_samples(x)
        mean = torch.mean(self.ensemble,axis = 0)
        return mean

class DeepEnsemble(Ensemble):
    def __init__(self,models, apply_softmax:bool = True, temperatura:int = 1):
        super().__init__(models)
        self.model = torch.nn.ParameterList()
        for m in models:
            self.model.append(m)
        self.apply_softmax = apply_softmax
        self.temperatura = temperatura
    
    def get_samples(self,x):
        ensemble = []
        for model in self.model:
            pred = model(x)
            pred /= self.temperatura
            if self.apply_softmax:
                pred = nn.functional.softmax(pred,dim=-1)
            ensemble.append(pred)
        self.ensemble = torch.stack(ensemble)
        return self.ensemble