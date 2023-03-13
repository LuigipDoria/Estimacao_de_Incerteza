import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

if torch.cuda.is_available():
    device = 'cuda'  
else:
    device = 'cpu'
print(device)

def train(epoch, model, loss_criterion, optimizer, scheduler,trainloader):
    model.train()
    model.to(device)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    scheduler.step()
    acc = 100.*correct/total
    return train_loss, acc

def test(epoch, model, loss_criterion, optimizer, scheduler, best_acc, best_loss,name,testloader):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100.*correct/total
    scheduler.step()
    
    if acc > best_acc:
        #print('Saving..')
        torch.save(model.state_dict(), ('./Resultado_parciais/{}_acc.pth').format(name))
        best_acc = acc
        best_loss = test_loss
    if acc == best_acc:
        if loss < best_loss:
            torch.save(model.state_dict(), ('./Resultado_parciais/{}_acc.pth').format(name))
            best_loss = test_loss
    
    return test_loss, acc,best_acc, best_loss

def testa_acuracia(model, testloader):
    # Calcula a acuracia da rede
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to("cuda"))
            outputs = outputs.to("cpu")
            outputs_numpy = outputs.to("cpu").numpy()

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    acuracia = round(100 * correct / total,3)
    return acuracia

def caculate_outputs(nets, images):
    outputs = list()
    for i in range(len(nets)):
        outputs.append(nets[i](images))
    return outputs

def calculate_correct(predicted, labels):
    correct_aux = (predicted == labels.to("cuda")).sum().item()
    return round(1-correct_aux/10000,4)

def calculate_predicted(mean_list):
    uncs_max, predicted_aux = torch.max(mean_list.data, 1)
    uncs_var = torch.var(mean_list,1)
    uncs_entr = torch.special.entr(mean_list)
    uncs_sum_entr = -torch.sum(uncs_entr, dim=-1)
    return uncs_max, uncs_var, uncs_sum_entr, predicted_aux