import torch


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

