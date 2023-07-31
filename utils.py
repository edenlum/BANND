import os
import torch


def save_gradient_means(gradients, labels, is_poisoned, save_dir='gradient_means'):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get the unique labels (classes)
    unique_labels = torch.unique(labels)

    # Initialize a dictionary to hold the mean gradients for each group
    mean_gradients = {label.item(): {} for label in unique_labels}
    mean_gradients = {}

    # Divide the gradients into groups and compute the mean for each group
    for name, grad_list in gradients.items():
        mean_gradients[name] = 
        # Divide the gradients into groups
        grad_groups = {label.item(): [] for label in unique_labels}
        grad_groups['poisoned'] = []
        for grad, label, poisoned in zip(grad_list, labels, is_poisoned):
            if poisoned:
                grad_groups['poisoned'].append(grad)
            else:
                grad_groups[label.item()].append(grad)

        # Compute the mean gradient for each group
        for group, grads in grad_groups.items():
            mean_gradients[group][name] = torch.stack(grads).mean(dim=0)

    # Save the mean gradients to disk
    for group, grads in mean_gradients.items():
        torch.save(grads, os.path.join(save_dir, f'mean_gradients_{group}.pt'))
