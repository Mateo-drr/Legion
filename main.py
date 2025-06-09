# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:12:08 2025

@author: mateo
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from consensus import GethConsensus
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import torch
import wandb
import copy
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from bitsandbytes.optim import Lion
import utils as ut

# PARAMS
# torch.set_num_threads(8)
# torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark = True

configD = {
    'runs':1,
    'lr': 1e-4,
    'ds2use': 0,
    'datasets': ['mnist', 'cifar100'],
    'num_epochs': 25,
    'batch': 512,
    'hidSize': 256,
    'dead':0.7, # Threshold for considering a neuron as dead [0;1] 
    'liveness':0.25, # A neuron has to be alive for >25% of the epochs to not be prunned
    'plateauWindow':5,
    'plateauTH':0.1,
    'gclip':2,
    'device':'cuda',
    'wb': False,
    'save':False,
    'project_name': 'Legion',
    'basePath': Path(__file__).resolve().parent,  # base dir
    'modelDir': Path(__file__).resolve().parent / 'weights',
}
config = SimpleNamespace(**configD)

#GLOBAL VARS
neuronStates=[]



# Make sure modelDir exists
config.modelDir.mkdir(parents=True, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(),
    transforms.RandomPerspective(),
    #transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

if config.datasets[config.ds2use] == 'mnist':
    # Load MNIST datasets
    train_ds = datasets.MNIST(root=config.basePath / 'data', train=True, download=True, transform=transform)
    valid_ds = datasets.MNIST(root=config.basePath / 'data', train=False, download=True, transform=transform)
elif config.datasets[config.ds2use] == 'cifar100':
    #Load cifar100
    train_ds = datasets.CIFAR10(root=config.basePath / 'data', train=True, download=True, transform=transform)
    valid_ds = datasets.CIFAR10(root=config.basePath / 'data', train=False, download=True, transform=transform)
    
train_dl = DataLoader(train_ds, batch_size=config.batch, shuffle=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=config.batch, pin_memory=True)        

    
def runModel(model):
    
    metrics = [0]
    dead = []
    prunned = []
    # optimizer = Lion(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss().to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    
    # Initialize Weights & Biases
    if config.wb:
        wandb.init(project=config.project_name, config=configD)
    
    bestTloss = float('inf')
    bestVloss = float('inf')
    lastLoss = float('inf')
    pastVlosses = []
    
    scaler = torch.amp.GradScaler(device='cuda')
    
    for epoch in tqdm(range(config.num_epochs), desc="Epoch"):
        
        ###
        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        # Print the total number of parameters
        print(f'\nTotal number of parameters: {total_params}')
        # Initialize the epoch neuron state dictionary
        neuronStates.append({'alive':None,
                             'dead':None,
                             'prunned':None
                             })
        
        ###
        
        model.train()
        trainLoss = 0
    
        iterab = train_dl    
    
        for images, labels in iterab:
            images, labels = images.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
    
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
            
            scaler.scale(loss).backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gclip)
            scaler.step(optimizer)         # Optimizer step
            scaler.update()                # Update scaler for next iteration
    
            trainLoss += loss.item()
            if config.wb:
                wandb.log({"TLoss": loss.item(), 'Learning Rate': optimizer.param_groups[0]['lr']})
    
    
        avg_loss = trainLoss / len(train_dl)
        print(f'Epoch {epoch+1}, Training Loss: {avg_loss}')
    
        model.eval()
        validLoss = 0
        all_preds = []
        all_labels = []
        reluMask = []
        
        with torch.no_grad():
            
            iterab = valid_dl    
            
            for images, labels in iterab:
                images, labels = images.to(config.device), labels.to(config.device)
        
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
        
                validLoss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                reluMask.append(model.relu_mask.cpu().numpy())
        
        conf_mat = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Validation Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")
        
        # Metrics Calculation
        avg_lossV = validLoss / len(valid_dl)
        print(f'\nEpoch {epoch+1}, Validation Loss: {avg_lossV}')
        
        if avg_loss <= bestTloss and avg_lossV <= bestVloss:
            bestModel = copy.deepcopy(model)
            bestTloss = avg_loss
            bestVloss = avg_lossV
            if config.save:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'losstv': (avg_loss, avg_lossV),
                    'config': config,
                }, config.modelDir / 'best.pth')
                
            metrics[0] = (conf_mat,acc,prec,recall,f1)
            
            # plot_confusion_matrix(all_labels, all_preds, class_names=[str(i) for i in range(10)])
        
        
        '''
        Neuron analysis
        '''
        
        #Concatenate the activation masks of all the batches
        reluMask = np.concatenate(reluMask, axis=0)
        #plot them
        ut.plotNeuronActivation(reluMask)
        
        # currently dead & alive neurons
        alive,d=ut.neuronSweeper(reluMask,config)
        # dead.append(d.tolist())
        neuronStates[epoch]['dead'] = d.tolist()
        neuronStates[epoch]['alive'] = alive.tolist()
        
        print('Loss comparison:', avg_lossV, 'vs last', lastLoss)
        
        #TODO Revive neurons on plateau
        
        # pastVlosses.append(avg_lossV)
        # if len(pastVlosses) > config.plateauWindow and np.mean(np.diff(pastVlosses)) < config.plateauTH:
        #     print(f'Reviving!! {pruneCheck(dead[-1], dead, getRevived=True)}')
        #     print(pastVlosses, np.mean(np.diff(pastVlosses)))
        #     model = GethConsensus(connectivity=alive, weights=learnedWeights).to(config.device)
        #     optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        
        #Prune dead neurons
        if avg_lossV < lastLoss and ut.pruneCheck(neuronStates):
            #pruneCheck works only after the first epoch has passed
            
            lastLoss = avg_lossV
            
            #check which neurons to prune
            alive, toPrune = ut.choosePrune(neuronStates, config) #note this alive list is not stored!
            
            #Avoid re init model if new toPrune is same as old!
            if toPrune != neuronStates[epoch-1]['prunned']:
                
                print('Prunning!!')
                
                learnedWeights = model.state_dict()
                model = GethConsensus(connectivity=torch.tensor(alive),
                                      weights=learnedWeights).to(config.device)
                optimizer = optim.AdamW(model.parameters(), lr=config.lr)
            else:
                print('Liveness check showed no new neurons to prune, skipping')

                model.disable_pruning_hooks()
                
            neuronStates[epoch]['prunned'] = toPrune
        
        #Halt prunning to stabilize training
        else:
            #currently prunned neurons didn't change
            print('Skipping prunning...')
            if epoch > 0:
                neuronStates[epoch]['prunned'] = neuronStates[epoch-1]['prunned']
            model.disable_pruning_hooks()
            
        print(f'Dead: {neuronStates[-1]["dead"]}')
        print('Prunned:', end=' ') 
        for i in range(len(neuronStates)):
            print(neuronStates[i]['prunned']) 
        
        if config.wb:
            wandb.log({
                "Validation Loss": avg_lossV,
                "Training Loss": avg_loss,
                "Validation Accuracy": acc,
                "Validation Precision": prec,
                "Validation Recall": recall,
                "Validation F1": f1
            })
    
    if config.wb:
        wandb.finish()
        
    return metrics




'''
Run models
'''

#calculate patch variables assuming square images
C,H,W = train_ds[0][0].shape
print('Images shape',C,'x', H,'x',W,)

resA = []
for i in tqdm(range(0,config.runs)):
    model = GethConsensus().to(config.device)
    resA.append(runModel(model))
    
ut.summarize_results(resA)