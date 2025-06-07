# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:12:08 2025

@author: mateo
"""

import torch
import torch.nn as nn
import sparselinear as sl


#%%
'''
import torch
import sparselinear as sl

# Define connectivity tensor for neurons 1 and 3
# Each neuron connects to all 5 input features
# Shape: (2, nnz), where nnz = 10 (5 inputs for neuron 1 + 5 inputs for neuron 3)
alive=[False,True,False,True,False]
connectivity = torch.LongTensor([
    [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],  # Output neuron indices (1 and 3)
    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]   # Input feature indices
])

# Create SparseLinear layer
layer = sl.SparseLinear(
    in_features=5,
    out_features=5,
    connectivity=connectivity,
    sparsity=0.0,  # Normally dense, but connectivity should override
    bias=True
)

#set bias of dead neurons to 0
with torch.no_grad():
    for i,neuron in enumerate(alive):
        if not neuron:
            layer.bias[i] = 0
        


# Sample input
input = torch.randn(2, 5)  # Batch size of 2, 5 input features
print("Input:\n", input)

# Forward pass
output = layer(input)
print("\nOutput shape:", output.shape)
print("Output (expect non-zero only at indices 1 and 3):\n", output)

print('weights', layer.weight)
print('bias', layer.bias)

# Inspect weights
print("\nWeight tensor indices:\n", layer.weight.indices())
print("Weight tensor values:\n", layer.weight.values())
print("Bias:\n", layer.bias)

# Verify sparsity
nonzero_weights = layer.weight.values().numel()
total_possible_weights = 5 * 5
print(f"\nNon-zero weights: {nonzero_weights}")
print(f"Effective sparsity: {(total_possible_weights - nonzero_weights) / total_possible_weights:.2f}")

# Simple training loop
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
target = torch.zeros(2, 5)
target[:, [1, 3]] = torch.randn(2, 2)  # Non-zero targets for neurons 1 and 3
print("\nTarget:\n", target)

for epoch in range(3):
    optimizer.zero_grad()
    output = layer(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
#'''
#%%

class GethConsensus(nn.Module):
    def __init__(self, inSize=784, hidSize=256,
                 connectivity=None, weights=None,
                 hook=True):
        super(GethConsensus, self).__init__()
        
        # self.fc1 = nn.Linear(784, 256)  
        self.fc2 = nn.Linear(hidSize, 10)  
        
        if connectivity is not None:
            
            #get bool tensor of which neurons are alive
            alive = torch.zeros(hidSize, dtype=torch.bool)
            alive[connectivity] = True
            self.alive = alive
            
            # Format the connectivity matrix 
            rows = connectivity.repeat_interleave(inSize)  # shape: [len(alive_neurons) * input_size]
            cols = torch.tile(torch.arange(inSize), (len(connectivity),))  # same shape as rows
            connectivity = torch.stack([rows, cols])
        
        if connectivity is None:
            row = torch.arange(hidSize).repeat_interleave(inSize)
            col = torch.arange(inSize).repeat(hidSize)
            connectivity = torch.stack([row, col])
        
        self.connectivity=connectivity
        
                
        self.sparse = sl.SparseLinear(in_features=inSize,
                                      out_features=hidSize,
                                      connectivity=connectivity,
                                      sparsity=0.0,
                                      bias=True)
        
        self._initialize_weights(weights)
        
        self.relu = nn.ReLU(inplace=True)
        self.relu_mask = None  # will store mask for active neurons
        self.hook = hook
        self._registered_hooks = []  # Track registered hooks
    
    def forward(self, x):
        
        x = torch.flatten(x,start_dim=1)
        

        #run the sparse linear even if fully connected
        x = self.sparse(x) 
        
        x = self.relu(x)
        
        
        
        # Save mask where ReLU was active (non-zero)
        self.relu_mask = (x != 0).float().detach()

        # Register backward hook to zero out gradients for inactive neurons
        if self.training and self.hook:
            handle = x.register_hook(self._hook_zero_inactive)
            self._registered_hooks.append(handle)

        x = self.fc2(x)
        
        return x
    
    def _clear_hooks(self):
        """Remove all registered hooks"""
        for handle in self._registered_hooks:
            handle.remove()
        self._registered_hooks.clear()
    
    def enable_pruning_hooks(self):
        """Enable gradient masking for pruning"""
        self.hook = True
    
    def disable_pruning_hooks(self):
        """Disable gradient masking to allow recovery"""
        self.hook = False
        self._clear_hooks()  # Immediately clear any existing hooks
    
    def _hook_zero_inactive(self, grad):
        # Ensure mask matches the dtype and device of grad
        mask = self.relu_mask.to(dtype=grad.dtype, device=grad.device)
        return grad * mask
    
    def _initialize_weights(self, weights):
        if weights is None:
            # Simple He/Kaiming initialization for all parameters
            for name, param in self.named_parameters():
                if 'weight' in name:
                    if param.dim() >= 2:  # Only apply kaiming to 2D+ tensors
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    else:  # For 1D tensors (sparse values), use normal with He scaling
                        std = (2.0 / param.size(0)) ** 0.5
                        nn.init.normal_(param, mean=0.0, std=std)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.01)  # Small positive bias for ReLU
            return
        
        with torch.no_grad():
            # 1. Load sparse layer weights
            if 'sparse.weights' in weights and 'sparse.indices' in weights:
                old_values = weights['sparse.weights']  # Note: 'weights' not 'weight'
                old_indices = weights['sparse.indices']
                
                # Get current sparse weight structure
                current_sparse = self.sparse.weight
                current_indices = current_sparse.indices()
                current_values = current_sparse.values()
                
                # Create a mapping from (out_idx, in_idx) to value for old weights
                old_weight_map = {}
                for i in range(old_indices.shape[1]):
                    out_idx, in_idx = old_indices[0, i].item(), old_indices[1, i].item()
                    old_weight_map[(out_idx, in_idx)] = old_values[i].item()
                
                # Fill new values where indices match
                new_values = current_values.clone()
                for j in range(current_indices.shape[1]):
                    out_idx, in_idx = current_indices[0, j].item(), current_indices[1, j].item()
                    if (out_idx, in_idx) in old_weight_map:
                        new_values[j] = old_weight_map[(out_idx, in_idx)]
                
                # Update the sparse tensor values
                current_sparse._values().copy_(new_values)
            
            # 2. Load sparse layer biases
            if 'sparse.bias' in weights and hasattr(self, 'alive'):
                old_bias = weights['sparse.bias']
                current_bias = self.sparse.bias
                
                # Map old bias values to current neurons based on alive mask
                old_idx = 0
                for i in range(len(current_bias)):
                    if i < len(self.alive) and self.alive[i]:
                        if old_idx < len(old_bias):
                            current_bias[i] = old_bias[old_idx]
                            old_idx += 1
                    else:
                        current_bias[i] = 0.0
            
            # 3. Load fc2 weights and biases directly (assuming no pruning in fc2)
            if 'fc2.weight' in weights:
                self.fc2.weight.data.copy_(weights['fc2.weight'])
            
            if 'fc2.bias' in weights:
                self.fc2.bias.data.copy_(weights['fc2.bias'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    