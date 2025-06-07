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
    plt.imshow(reluMask, aspect='auto', cmap='gray_r')  # white=active, black=dead
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
    return torch.tensor(alive), dead

def pruneCheck(current_dead, previous_dead_list, prunned_list, getRevived=False):
    """
    Check if current dead neurons are a superset of previously dead neurons.
    Only allow pruning if we're killing the same neurons as before + potentially new ones.
    
    Args:
        current_dead: numpy array of currently dead neuron indices
        previous_dead_list: list of previous dead neuron arrays
    
    Returns:
        bool: True if safe to prune, False otherwise
    """
    
    current_dead = neuronStates[-1]['dead']
    previous_dead_list = [neuronStates[i]['dead'] for i in range(len(neuronStates))]
    prunned_list = [neuronStates[i]['prunned'] for i in range(len(neuronStates))]
    
    if len(previous_dead_list) == 1 and len(current_dead) != 0:
        return True  # First pruning is always safe
    
    # Get the most recent dead neurons
    last_dead = previous_dead_list[-2]
    
    # Convert to sets for easier comparison
    current_dead_set = set(current_dead)
    last_dead_set = set(last_dead)
    
    # Check if all previously dead neurons are still dead
    previously_dead_still_dead = last_dead_set.issubset(current_dead_set)
    
    if not previously_dead_still_dead:
        print(f"⚠️  Previously dead neurons {last_dead_set - current_dead_set} are now alive!")
        print("Skipping pruning to avoid weight resurrection")
        if getRevived:
            return last_dead_set - current_dead_set
        else:
            return False
    
    # Check if we have new dead neurons (actual progress)
    current_prunned_set = set(prunned_list[-1])
    new_dead_neurons = current_dead_set - last_dead_set
    if len(new_dead_neurons) == 0 and len(current_dead_set - current_prunned_set) == 0 :
        print("No new dead neurons found, skipping pruning")
        return False
    
    if len(new_dead_neurons) != 0:
        print(f"✅ Safe to prune. New dead neurons: {new_dead_neurons}")
    else:
        print(f"✅ Safe to prune. Dead neurons to prune: {current_dead_set - current_prunned_set}")
    return True