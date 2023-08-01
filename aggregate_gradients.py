import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

def aggregate_gradients_cosine(grads_list, labels, is_poisoned):
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
            cosine_similarities = F.cosine_similarity(torch.flatten(c_grads, start_dim=1), torch.flatten(c_mean, start_dim=1))
            c_weights = torch.softmax(cosine_similarities, dim=0)
            weights.append(c_weights)

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(torch.flatten(c_grads, start_dim=1).cpu())
            fig, ax = plt.subplots()

            # plot with colors representing classes
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=is_poisoned[(labels==c).cpu()], cmap='tab10')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('PCA of Tensors')

            # create legend
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)

            plt.savefig(f'plots/{name}_{c}.png')
        

        grads = torch.cat(grads)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(torch.flatten(grads, start_dim=1).cpu())
        fig, ax = plt.subplots()

        # plot with colors representing classes
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=is_poisoned, cmap='tab10')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('PCA of Tensors')

        # create legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)

        plt.savefig(f'plots/{name}.png')
        
        print("poisoned", torch.norm(torch.flatten(grads[is_poisoned], start_dim=1), dim=1).mean().item())
        print("not pois", torch.norm(torch.flatten(grads[~is_poisoned], start_dim=1), dim=1).mean().item())
        print("Poisoned grads:", grads[is_poisoned])
        print("Not Poisoned grads:", grads[~is_poisoned])
        weights = torch.cat(weights)
        mean = torch.einsum('i,i...->...', weights, grads)
        aggregated_gradients[name] = mean

    return aggregated_gradients