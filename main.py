# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch,torchvision
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
from mne.decoding import CSP # Common Spatial Pattern Filtering
from bcidataset import bcidataset,bcinet
import matplotlib as plt
import numpy as np
def eval(net,optimizer,lr,criterion,testloader,device):
    net.eval()
    correct = 0
    false = 0
    TP,TN,FP,FN = 0,0,0,0
    pre = []
    labels = []
    for i, (data, label) in enumerate(testloader):

        data = data.to(torch.float32)
        data = data.to(device)
        label = label.to(torch.float32)

        label = label.to(device)

        output = net(data)
        _,predicted  = torch.max(output,1)
        _,label_dd =  torch.max(label,1)
        pre.append(predicted)
        labels.append(label_dd)
        correct += (predicted == label_dd).sum().item()
        false += (predicted != label_dd).sum().item()
        TP += (predicted == 1 and label_dd ==1).sum().item()
        TN += (predicted ==0 and label_dd == 0).sum().item()
        FN += (predicted == 0and label_dd ==1).sum().item()
        FP +=  (predicted == 1and label_dd ==0).sum().item()


    precision = TP*100/(TP+FP)
    Recall = TP*100/(TP+FN)
    acc = correct/(len(testloader))*100
    f1_s = 2*precision*Recall/(precision+Recall)

    print("F1-Score:{:.4f}".format(f1_s))
    print("accuracy = {%d}%% " %(acc))
    print("precision = {%d}%% " %(precision))
    print("Recall = {%d}%%" %(Recall))





def main():
    filename = "C:/Users/86181/Desktop/ugPRML_EEGProject/Data"
    bci_Dataset = bcidataset(os.path.join(filename,"data"))
    net = bcinet(22)

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(device)
    net.to(device)
    '''
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.uniform_(layer.weight)
            layer.bias.data.fill_(1)
    '''
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),0.001, momentum=0.9,weight_decay=0.0)
    lambda1 = lambda x: 0.95
    lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda1,last_epoch=-1,verbose=True)

    train_size = int(0.83 * len(bci_Dataset))
    test_size = len(bci_Dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(bci_Dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    testloader = DataLoader(test_dataset,batch_size=1,shuffle=True)
    acc = []
    loss = 0

    #net.load_state_dict(torch.load(os.path.join(filename,"ckpoint5.pth"))["net"])
    best_loss = 0.5
    for epoch in range(80):
        correct = 0
        running_loss = 0
        for i,(data,label) in enumerate(trainloader):


            optimizer.zero_grad()
            data = data.to(torch.float32)
            data = data.to(device)
            label =label.to(torch.float32)
            label = label.to(device)
            output = net(data)

            regularization_loss = 0
            for param in net.parameters():
                regularization_loss += torch.sum(abs(param))

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss +=loss.item()


            if i % 10 == 9 :
                print('epoch: %d, iter:%5d \t loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

            _, predicted = torch.max(output, 1)
            _, label_dd = torch.max(label, 1)


            correct += (predicted == label_dd).sum().item()
        lr_scheduler.step()
        print("accuracy = {%d}%% of %d " % (correct / (32*len(trainloader)) * 100,32*len(trainloader)))
        acc.append(correct / (4*len(trainloader)))
        if epoch %5 == 0 and loss.item() < best_loss:
            torch.save({"net":net.state_dict(), 'optimizer':optimizer.state_dict(),'epoch':epoch},os.path.join(filename,"ckpoint5.pth"))
    eval(net,optimizer,optimizer.param_groups[0]['lr'],criterion,testloader,device)

if __name__ == '__main__':
    main()






