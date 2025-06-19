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
import numpy as np
from bitsandbytes.optim import Lion
import utils as ut

# PARAMS
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark = True

configD = {
    'runs':1,
    'lr': 7e-4,
    'ds2use': 0,
    'datasets': ['mnist', 'cifar100'],
    'num_epochs': 200,
    'batch': 512,
    'hidSize': 256,
    #PARAMS for controlling pruning
    'dead':0.4, # Threshold for considering a neuron dead (0) or alive (1) 
    'liveness':0.6, # A neuron has to be alive for >25% of the epochs to not be pruned
    'killLim': 10, # Maximum number of neurons that can be pruned per epoch
    'plateauWindow': 5, # Number of epochs to be considered as plateau
    'plateauTH': 0.1, # Minimum difference in vLoss for liveness to be reduced
    'livenessLim': 0.5, #
    #plateau
    'gclip': 2,
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
    pruned = []
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
    
    for epoch in range(config.num_epochs): #tqdm(range(config.num_epochs), desc="Epoch"):
        
        ###
        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        # Print the total number of parameters
        print(f'\nTotal number of parameters: {total_params}')
        # Initialize the epoch neuron state dictionary
        neuronStates.append({'alive':None,
                             'dead':None,
                             'pruned':None,
                             'loss':{'train':None,
                                     'valid':None}
                             })
        
        ###
        
        model.train()
        trainLoss = 0
    
        iterab = train_dl    
    
        for images, labels in iterab:
            images, labels = images.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
    
            with torch.amp.autocast(device_type='cuda'):
                outputs, latent = model(images)  # Forward pass
                entropy = ut.EntropyLoss(latent)
                loss = criterion(outputs, labels) + entropy # Compute loss
            
            scaler.scale(loss).backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gclip)
            scaler.step(optimizer)         # Optimizer step
            scaler.update()                # Update scaler for next iteration
    
            trainLoss += loss.item()
            if config.wb:
                wandb.log({"TLoss": loss.item(), 'Learning Rate': optimizer.param_groups[0]['lr']})
    
    
        avg_loss = trainLoss / len(train_dl)
        neuronStates[epoch]['loss']['train'] = avg_loss
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
                    outputs, latent = model(images)
                    entropy = ut.EntropyLoss(latent)
                    loss = criterion(outputs, labels) + entropy
        
                validLoss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                #get the reluMask of each layer... for now only one layer
                reluMask.append(model.getReluMasks()['msparse'].cpu().numpy())
        
        conf_mat = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Validation Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # Metrics Calculation
        avg_lossV = validLoss / len(valid_dl)
        neuronStates[epoch]['loss']['valid'] = avg_lossV
        print(f'Epoch {epoch+1}, Validation Loss: {avg_lossV}')
        
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
        
        
        
        pastVlosses.append(avg_lossV)
        
        #Reduce chance of further pruning | revive neurons
        if len(pastVlosses) > config.plateauWindow and np.mean(np.abs(np.diff(pastVlosses[-config.plateauWindow:]))) < config.plateauTH:
            print(pastVlosses[-config.plateauWindow:], np.mean(np.abs(np.diff(pastVlosses[-config.plateauWindow:]))))
            
            config.liveness = config.liveness*0.75
            config.plateauWindow = config.plateauWindow*2
            config.plateauTH = config.plateauTH*0.75
            print('reducing liveness th. New params', config.plateauWindow, config.plateauTH)
            alive, toPrune = ut.choosePrune(neuronStates, config, thUpdate=True)
            
            
            if set(toPrune) != set(neuronStates[epoch-1]['pruned']):
                learnedWeights = model.state_dict()
                model = GethConsensus(connectivity=torch.tensor(alive),
                                      weights=learnedWeights).to(config.device)
                optimizer = optim.AdamW(model.parameters(), lr=config.lr)
                
            neuronStates[epoch]['pruned'] = toPrune #should be the same if sets are equal...
            
            #Check if the config.plateauWindow is achievable within epoch limits
            #Using this check as a trigger for adding a new layer
            if config.num_epochs - epoch < config.plateauWindow:
                print('Evolving a new hidden layer')
                hidSize = len(neuronStates[epoch]['pruned'])
                #TODO
        
        #Prune dead neurons
        elif avg_lossV < lastLoss and ut.pruneCheck(neuronStates):
            #pruneCheck works only after the first epoch has passed
            
            lastLoss = avg_lossV
            
            #check which neurons to prune
            alive, toPrune = ut.choosePrune(neuronStates, config) #note this alive list is not stored!
            
            #Avoid re init model if new toPrune is same as old!
            if toPrune != neuronStates[epoch-1]['pruned'] and len(toPrune) != 0:
                
                print('Prunning!!')
                
                learnedWeights = model.state_dict()
                model = GethConsensus(connectivity=torch.tensor(alive),
                                      weights=learnedWeights).to(config.device)
                optimizer = optim.AdamW(model.parameters(), lr=config.lr)
            else:
                print('Liveness check showed no new neurons to prune, skipping')
                
            neuronStates[epoch]['pruned'] = toPrune
        
        #Halt prunning to stabilize training
        else:
            #currently pruned neurons didn't change
            print('Skipping prunning...')
            if epoch > 0:
                neuronStates[epoch]['pruned'] = neuronStates[epoch-1]['pruned']
                
        
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