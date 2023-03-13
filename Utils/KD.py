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

softmax_op = nn.Softmax(dim=1)
mseloss_fn = nn.MSELoss()

def hparamToString(hparam):
    """
    Convert hparam dictionary to string with deterministic order of attribute of hparam in output string
    """
    hparam_str = ''
    for k, v in sorted(hparam.items()):
        hparam_str += k + '=' + str(v) + ', '
    return hparam_str[:-2]

def hparamDictToTuple(hparam):
    """
    Convert hparam dictionary to tuple with deterministic order of attribute of hparam in output tuple
    """
    hparam_tuple = [v for k, v in sorted(hparam.items())]
    return tuple(hparam_tuple)

def studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha):
    """
    One training step of student network: forward prop + backprop + update parameters
    Return: (loss, accuracy) of current batch
    """
    optimizer.zero_grad()
    teacher_pred = None
    if (alpha > 0):
        with torch.no_grad():
            teacher_pred = teacher_net(X)
    student_pred = student_net(X)
    loss = studentLossFn(teacher_pred, student_pred, y, T, alpha, teacher_net)
    loss.backward()
    optimizer.step()
    accuracy = float(torch.sum(torch.argmax(student_pred, dim=1) == y).item()) / y.shape[0]
    
    return loss, accuracy

def trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs, name,
    train_loader, val_loader, 
    print_every=0, 
    fast_device=torch.device('cuda')):
    """
    Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch
    Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
    """
    teacher_net.eval()
    train_loss_list, train_acc_list, val_acc_list = [], [], []
    T = hparam['T']
    alpha = hparam['alpha']
    student_net.dropout_input = hparam['dropout_input']
    student_net.dropout_hidden = hparam['dropout_hidden']
    optimizer = optim.SGD(student_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])

    def studentLossFn(teacher_pred, student_pred, y, T, alpha,teacher_net):
        """
        Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
        Return: loss
        """
        if (alpha > 0):
            if str(type(teacher_net)) == "<class '__main__.DeepEnsemble'>":
                loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), teacher_pred, reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (1 - alpha)
            else:
                loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (1 - alpha)
        else:
            loss = F.cross_entropy(student_pred, y)
        return loss
    
    best_acc = 0
    best_loss = 100
    for epoch in range(num_epochs):
        if epoch == 0:
            if val_loader is not None:
                _, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
                val_acc_list.append(val_acc)
                print('epoch: %d validation accuracy: %.3f' %(epoch, val_acc))
        for i, data in enumerate(train_loader, 0):
            X, y = data
            X, y = X.to(fast_device), y.to(fast_device)
            loss, acc = studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha)
            train_loss_list.append(loss)
            train_acc_list.append(acc)               
            if print_every > 0 and i % print_every == print_every - 1:
                if acc > best_acc:
                    best_acc_ = acc
                print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f best train accuracy: %.3f' %
                (epoch + 1, i + 1, len(train_loader), loss, acc, best_acc_))
        lr_scheduler.step()
        if acc > best_acc:
            torch.save(student_net.state_dict(), ('/home/luigi-doria/IC/Resultado_parciais/student{}_acc.pth').format(name))
            best_acc = acc
            best_loss = loss
        if acc == best_acc:
            if loss < best_loss:
                torch.save(student_net.state_dict(), ('/home/luigi-doria/IC/Resultado_parciais/student{}_acc.pth').format(name))
                best_loss = loss
        
        if val_loader is not None:
            _, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
            val_acc_list.append(val_acc)
            print('epoch: %d validation accuracy: %.3f' %(epoch + 1, val_acc))
    return {'train_loss': train_loss_list, 
            'train_acc': train_acc_list, 
            'val_acc': val_acc_list}


def my_loss(scores, targets, target,temperature = 5):
    soft_pred = softmax_op(scores / temperature)
    with torch.no_grad():
        if str(type(target)) == "<class '__main__.DeepEnsemble'>":
            soft_targets = targets
        else:
            soft_targets = softmax_op(targets / temperature)
    loss = mseloss_fn(soft_pred, soft_targets)
    return loss

def train_knowledge_distillation(epoch, aprendiz, optimizer, scheduler,target, temperature,trainloader):
    aprendiz.train()
    train_loss = 0
    correct = 0
    total = 0
    target.eval()
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        scores = aprendiz(images)
        with torch.no_grad():
            targets = target(images.to("cuda"))
        loss = my_loss(scores, targets, target,temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = scores.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()
    
    scheduler.step()
    acc = 100.*correct/total

    return train_loss, acc
        
def test_knowledge_distillation(epoch, aprendiz, optimizer, scheduler,target, temperature, best_acc, best_loss,name,testloader):
    aprendiz.eval()
    target.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = aprendiz(inputs)
            targets = target(inputs)
            loss = my_loss(outputs, targets, target,temperature)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    acc = 100.*correct/total
    if acc > best_acc:
        #print('Saving..')
        torch.save(aprendiz.state_dict(), ('/home/luigi-doria/IC/Resultado_parciais/student{}_acc.pth').format(name))
        best_acc = acc
        best_loss = test_loss
    if acc == best_acc:
        if loss < best_loss:
            torch.save(aprendiz.state_dict(), ('/home/luigi-doria/IC/Resultado_parciais/student{}_acc.pth').format(name))
            best_loss = test_loss
    return test_loss, acc, best_acc, best_loss

def knowledge_distillation(epoch, aprendiz, optimizer, scheduler,target, temperature, best_acc, best_loss, name,trainloader,testloader):
    loss_train, acc_train = train_knowledge_distillation(epoch, aprendiz, optimizer, scheduler,target, temperature,trainloader)
    loss_test , acc_test, best_acc, best_loss  = test_knowledge_distillation(epoch, aprendiz, optimizer, scheduler,target, temperature, best_acc, best_loss,name,testloader)
    print("Loss Train: {} || Acc Train: {} || Loss Teste: {} || Acc Teste: {} || Best Acc Teste: {}".format(loss_train, acc_train, loss_test, acc_test, best_acc))
    return best_acc, best_loss