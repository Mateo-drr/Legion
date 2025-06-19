# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 17:46:11 2025

@author: mateo
"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import math

#TODO fix this mess
ogSize = 0
    
def EntropyLoss(activations, alpha=0.2):
    """
    Encourage sparsity by penalizing high entropy 
    Low entropy = sparse (few active neurons)
    High entropy = dense (many active neurons)
    """
    global ogSize
    
    batch_size, current_size = activations.shape
    
    # Store original size the first time
    if ogSize == 0:
        ogSize = current_size
    
    # If pruned, pad with -inf to maintain original size
    if current_size < ogSize:
        pad_size = ogSize - current_size
        pad = torch.full((batch_size, pad_size), float('-inf'), device=activations.device)
        activations = torch.cat([activations, pad], dim=1)
     
    # Softmax and entropy calculation
    probs = torch.softmax(activations, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return torch.mean(entropy) * alpha

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn heatmap.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
def plotNeuronActivation(reluMask):
    plt.figure(figsize=(10, 5))
    plt.imshow(reluMask, aspect='auto', cmap='gray')  # white=active, black=dead
    plt.xlabel('Neurons')
    plt.ylabel('Samples')
    plt.title('Neuron activation mask after ReLU across validation samples')
    plt.colorbar(label='Active (1) vs Dead (0)')
    plt.show()
    
    
def summarize_results(results, class_names=[str(i) for i in range(10)], normalize=True):
    accs = [r[0][1] for r in results]
    precs = [r[0][2] for r in results]
    recalls = [r[0][3] for r in results]
    f1s = [r[0][4] for r in results]

    print(f"Average Accuracy: {np.mean(accs):.4f}")
    print(f"Average Precision: {np.mean(precs):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1 Score: {np.mean(f1s):.4f}")

    # Average Confusion Matrix
    conf_mats = [r[0][0] for r in results]
    avg_conf_mat = sum(conf_mats) / len(conf_mats)

    if normalize:
        avg_conf_mat = avg_conf_mat / avg_conf_mat.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_conf_mat, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
def neuronSweeper(reluMask,config):
    meanMask = reluMask.mean(axis=0)
    alive = np.where(meanMask > config.dead)[0]
    dead = np.where(meanMask <= config.dead)[0]
    print('Dead neurons', len(dead))
    return alive, dead

def deadMatrix(neuronStates,config):
    totalNeurons = len(neuronStates[0]['dead']) + len(neuronStates[0]['alive'])
    #create a one hot encoding list 1 for alive 0 for dead for each epoch
    matrix = np.zeros((len(neuronStates), totalNeurons), dtype=int)
    for epoch in range(len(neuronStates)):
        matrix[epoch, neuronStates[epoch]['alive']] = 1
        
        # Give pruned neurons a randomized second chance
        pruned_neurons = neuronStates[epoch]['pruned'] or []
        for neuron in pruned_neurons:
            if neuron < matrix.shape[1]:  # Safety check
                matrix[epoch, neuron] = np.random.uniform(config.liveness-0.1, config.liveness + 0.2)
        
    return matrix

# def choosePrune(neuronStates, config):
    
#     #Calcualte one-hot matrix of neuron states
#     matrix = deadMatrix(neuronStates)
    
#     #calcualte the average 'liveness' of the neurons along epochs
#     liveness = np.mean(matrix, axis=0)
    
#     toPrune = np.where(liveness <= config.liveness)[0]
#     alive = np.where(liveness > config.liveness)[0]
    
#     print(np.round(liveness,2))
    
#     if alive.tolist() != neuronStates[-1]['alive']:
#         print('neurons to be pruned changed!')
    
#     return alive, toPrune.tolist()

def plotPrune(losses,liveness,toPrune,config):
    # --- Plot 1: Loss evolution ---
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(losses, marker='o', label='Combined loss (train + valid)')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Highlight best epoch
    best_epoch = np.argmin(losses)
    plt.axvline(best_epoch, color='green', linestyle='--', label='Best Epoch')
    plt.legend()
    
    # --- Plot 2: Neuron Liveness ---
    plt.subplot(2, 1, 2)
    bar_width = 0.9  # or 1.0 for full width
    indices = np.arange(len(liveness))
    plt.bar(indices, liveness, width=bar_width, color='gray', label='Liveness')
    plt.axhline(config.liveness, color='red', linestyle='--', label='Prune Threshold')
    
    # Highlight neurons selected for pruning
    if len(toPrune) > 0:
        plt.bar(indices[toPrune], liveness[toPrune], width=bar_width, color='red', label='To Prune')
    
    # # Convert the list to a string for display
    # to_prune_str = ', '.join(map(str, toPrune)) if len(toPrune) > 0 else "None"
    
    # # Add the text below the plots
    # plt.figtext(0.5, -0.02, f"[{to_prune_str}]", 
    #             wrap=True, horizontalalignment='center', fontsize=10)
    for idx in toPrune:
        if liveness[idx] == 0:
            plt.plot(idx, 0, marker='v', color='red')
    
    plt.title("Weighted Liveness per Neuron")
    plt.xlabel("Neuron Index")
    plt.ylabel("Liveness")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def choosePrune(neuronStates, config, thUpdate=False):
    
    assert len(neuronStates) >= 2, 'Need at least 2 epochs of records'
    
    # Extract losses for ranking
    losses = []
    combined_loss = 0
    for state in neuronStates:
        if state['loss']['train'] is not None and not math.isnan(state['loss']['train']):
            combined_loss = state['loss']['train'] 
        else:
            combined_loss = float('inf')
        if state['loss']['valid'] is not None and not math.isnan(state['loss']['valid']):
            combined_loss += state['loss']['valid']
        else:
            combined_loss = float('inf')
            
    
        losses.append(combined_loss)
    
    losses = np.array(losses)
    # Rank-based weighting (0 for best/lowest loss, higher ranks for worse loss)
    ranks = np.argsort(np.argsort(losses))  # Double argsort gives ranks
    weights = 1.0 / (ranks + 1)  # Higher weight for lower ranks (better performance)
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    # print(f"Loss values: {np.round(losses, 4)}")
    # print(f"Ranks: {ranks}")
    # print(f"Weights: {np.round(weights, 4)}")
    
    
    matrix = deadMatrix(neuronStates, config)
    # if matrix.shape[0] < config.liveWindow:
    #     liveness = np.mean(matrix, axis=0)
    # else:
    #     liveness = np.mean(matrix[-config.liveWindow:], axis=0)
    
    # Calculate rank-weighted average liveness
    liveness = np.average(matrix, axis=0, weights=weights)

    # Neurons below threshold
    toPrune = np.where(liveness <= config.liveness)[0]

    pruned = neuronStates[-2]['pruned'] if neuronStates[-2]['pruned'] is not None else []
    if not thUpdate:
        if pruned is not None:
            # print(len(pruned), len(toPrune))
            limit = config.killLim + len(pruned)
            if config.killLim is not None and len(toPrune) > limit:
                sorted_toPrune = toPrune[np.argsort(liveness[toPrune])]
                toPrune = sorted_toPrune[:limit]
    else:
        # set the limit to the past amount of pruned neurons, or less (i.e. revive some)
        limit = len(pruned) if len(toPrune) > len(pruned) else len(toPrune)
        sorted_toPrune = toPrune[np.argsort(liveness[toPrune])]
        toPrune = sorted_toPrune[:limit]

    # Alive = all other neurons
    all_neurons = np.arange(len(liveness))
    alive = np.setdiff1d(all_neurons, toPrune)

    # print(np.round(liveness, 2))

    if alive.tolist() != neuronStates[-1]['alive']:
        print('neurons to be pruned changed!')
        
    plotPrune(losses, liveness, toPrune, config)

    return alive, toPrune.tolist()


def pruneCheck(neuronStates, getRevived=False):
    """
    Check if current dead neurons are a superset of previously dead neurons.
    Only allow pruning if we're killing the same neurons as before + potentially new ones.
    
    Args:
        current_dead: numpy array of currently dead neuron indices
        previous_dead_list: list of previous dead neuron arrays
    
    Returns:
        bool: True if safe to prune, False otherwise
    """
    
    if len(neuronStates) == 1:
        return False #avoid prunning on first epoch
    
    #TODO temporary fix
    current_dead = neuronStates[-1]['dead']
    previous_dead_list = [neuronStates[i]['dead'] for i in range(len(neuronStates))]
    
    pruned_list = []
    for i in range(len(neuronStates)):
        if neuronStates[i]['pruned'] is not None:
            pruned_list.append(neuronStates[i]['pruned'])
    
    # if len(previous_dead_list) == 1 and len(current_dead) != 0:
    #     return True  # First pruning is always safe
    
    # Get the most recent dead neurons
    last_dead = previous_dead_list[-2]
    
    # Convert to sets for easier comparison
    current_dead_set = set(current_dead)
    last_dead_set = set(last_dead)
    
    # Check if all previously dead neurons are still dead
    # previously_dead_still_dead = last_dead_set.issubset(current_dead_set)
    
    # if not previously_dead_still_dead:
    #     print(f"⚠️  Previously dead neurons {last_dead_set - current_dead_set} are now alive!")
    #     print("Skipping pruning to avoid weight resurrection")
    #     if getRevived:
    #         return last_dead_set - current_dead_set
    #     else:
    #         return False
    
    # Check if we have new dead neurons (actual progress)
    current_pruned_set = set(pruned_list[-1]) if len(pruned_list)!=0 else set([]) #empty set if no pruned neurons
    new_dead_neurons = current_dead_set - last_dead_set
    if len(new_dead_neurons) == 0 and len(current_dead_set - current_pruned_set) == 0 :
        print("No new dead neurons found, skipping pruning")
        return False
    
    if len(new_dead_neurons) != 0:
        print(f"✅ Safe to prune. New dead neurons: {new_dead_neurons}")
    else:
        print(f"✅ Safe to prune. Dead neurons to prune: {current_dead_set - current_pruned_set}")
    return True