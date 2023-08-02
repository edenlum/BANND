import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import *


def squash(vector):
    """
    Squashes a vector to have a norm between 0 and 1.

    Args:
    vector: A vector to squash.

    Returns:
    The squashed vector.
    """

    # Compute the norm of the vector
    norm = torch.norm(vector)

    # Squash the vector
    squashed_vector = norm * vector / (1 + norm**2)

    return squashed_vector


def aggregate_gradients(sample_gradients, iters=3, non_linearity=squash):
    """
    Aggregates gradients using a weighted average.

    Args:
    sample_gradients: A list of dictionaries containing the gradients for each parameter for each sample.

    Returns:
    A dictionary containing the aggregated gradients for each parameter.
    """
    
    # transform the list of dictionaries into a dictionary of lists
    sample_gradients = {name: [sample[name] for sample in sample_gradients] for name in sample_gradients[0].keys()}
    aggregated_gradients = {}
    for name, gradients in sample_gradients.items():
        sample_gradients[name] = torch.stack(gradients)
        b = torch.zeros(sample_gradients[name].shape[0]).to(sample_gradients[name].device)
        for i in range(iters):
            c = torch.softmax(b, dim=0)
            mean = torch.einsum('i,i...->...', c, sample_gradients[name])
            # squashed_mean = non_linearity(mean)
            # b = b + torch.sum((sample_gradients[name] * squashed_mean.unsqueeze(0)).view(c.shape[0], -1), dim=1)
            denom = torch.norm(sample_gradients[name].view(c.shape[0], -1), dim=1)
            b = b + torch.sum((sample_gradients[name] * mean.unsqueeze(0)).view(c.shape[0], -1), dim=1)/denom
            
        aggregated_gradients[name] = mean

    return aggregated_gradients


def aggregate_gradients_cosine(grads_list, labels, is_poisoned, similarity=F.cosine_similarity, plot=False):
    grads_dict = {name: [sample[name] for sample in grads_list] for name in grads_list[0].keys()}
    aggregated_gradients = {}
    for name, gradients in grads_dict.items():
        # seperate to classes
        grads = []
        weights = []
        for c in torch.unique(labels):
            if torch.stack(gradients)[labels==c].size(0) == 0:
                continue

            c_grads = torch.stack(gradients)[labels==c]
            grads.append(c_grads)
            c_mean = c_grads.mean(dim=0, keepdim=True)
            similarities = similarity(torch.flatten(c_grads, start_dim=1), torch.flatten(c_mean, start_dim=1))
            c_weights = torch.softmax(similarities, dim=0)
            weights.append(c_weights)

            if plot: plot_gradients_pca(c_grads, is_poisoned[(labels==c).cpu()], f"{name}_{c}")        

        grads = torch.cat(grads)
        if plot: plot_gradients_pca(grads, is_poisoned, name)

        weights = torch.cat(weights)
        print("Poisoned samples weights: ", weights[is_poisoned].mean()/weights.mean())
        mean = torch.einsum('i,i...->...', weights, grads)
        aggregated_gradients[name] = mean

    return aggregated_gradients



def aggregate_all_params(grads_list, labels, is_poisoned, similarity=F.cosine_similarity, normalize=True, plot=False, save_gradients=False, name_to_save="gradients_labels_poisoned"):
    """
    This function first combines all the gradients for all parameters into one big vector (for each sample).
    Then it computes the weight of each sample by comparing it to the mean of all samples.
    Then it computes the weighted average of the gradients.
    Finally, it splits the big vector back into the gradients for each parameter.
    """
    original_shapes = {name: grads_list[0][name].shape for name in grads_list[0].keys()} # shape of each parameter
    lengths = [g.nelement() for g in [grads_list[0][name] for name in grads_list[0].keys()]] # length of each gradient per parameter
    # tensor of shape (num_samples, sum(lengths))
    gradients = torch.stack([torch.cat([torch.flatten(sample[name], start_dim=0) for name in sample.keys()]) for sample in grads_list]) 
    if normalize: 
        gradients = (gradients - gradients.mean(dim=0))/gradients.std(dim=0)
    if save_gradients:
        save_gradients_labels_poisoned(name_to_save, gradients, labels, is_poisoned)

    # seperate to classes
    grads = []
    weights = []
    for c in torch.unique(labels):
        if gradients[labels==c].size(0) == 0: # if there are no samples of this class
            continue

        c_grads = gradients[labels==c]
        grads.append(c_grads)
        c_mean = c_grads.mean(dim=0, keepdim=True)
        similarities = similarity(torch.flatten(c_grads, start_dim=1), torch.flatten(c_mean, start_dim=1))
        c_weights = torch.softmax(similarities, dim=0)
        weights.append(c_weights)

        if plot: plot_gradients_pca(c_grads, is_poisoned[(labels==c).cpu()], f"all_params_{c}")

    grads = torch.cat(grads)
    if plot: plot_gradients_pca(grads, is_poisoned, "all_params")

    weights = torch.cat(weights)
    avg_w_poisoned = weights[is_poisoned].mean()*len(weights)
    print("Poisoned samples weights average: ", avg_w_poisoned)
    mean = torch.einsum('i,i...->...', weights, grads)

    aggregated_gradients = {}
    mean_per_param = torch.split(mean, lengths)
    for i, name in enumerate(grads_list[0].keys()):
        aggregated_gradients[name] = mean_per_param[i].reshape(original_shapes[name])

    return aggregated_gradients, avg_w_poisoned


def plot_gradients_pca(gradients, is_poisoned, name):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(torch.flatten(gradients, start_dim=1).cpu())
    fig, ax = plt.subplots()

    # plot with colors representing classes
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=is_poisoned, cmap='tab10')

    # Compute the mean of all the gradients
    mean_all = pca_result.mean(axis=0)

    # Compute the mean of the gradients that are not poisoned
    mean_not_poisoned = pca_result[~is_poisoned].mean(axis=0)

    # Plot the means
    ax.scatter(mean_all[0], mean_all[1], c='red', marker='x', label='Mean of All Gradients')
    ax.scatter(mean_not_poisoned[0], mean_not_poisoned[1], c='green', marker='x', label='Mean of Not Poisoned Gradients')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA of Tensors')

    # create legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.legend(loc='best')  # Add legend for the means

    plt.savefig(f'plots/{name}.png')
    plt.close()

