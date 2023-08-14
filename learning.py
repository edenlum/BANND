import torch
from torch import nn
from tqdm import tqdm

import settings
from aggregate_gradients import aggregate_all_params
from utils import save_model, save_stats_plots


def calc_accuracy(device, model, data_loader):
    """
    Calculate the model's accuracy on the data, i.e., the percentage of successfully
    predicted samples
    """

    correct = 0
    total = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train(
    device,
    model,
    train_loader,
    should_save_model=False,
    model_file_name=None,
    should_save_stats=False,
    stats_file_name=None,
    test_loader_clean=None,
    test_loader_poisoned=None,
    calc_states_every_nth_iter=10,
    epochs=1,
    defend=False,
):
    print("training model...")

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(**({"reduction": "none"} if defend else {}))

    accuracies = []
    attack_success_rates = []

    for _ in tqdm(range(epochs), desc="epoch"):
        for i, batch in enumerate(tqdm(train_loader, desc="batch")):
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if defend:
                defense(loss, optimizer, model, labels)
            else:
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if should_save_stats and i % calc_states_every_nth_iter == 0:
                # compute accuracy on clean test dataset
                accuracy = calc_accuracy(device, model, test_loader_clean)
                accuracies.append(accuracy)
                # compute attack success rate on poisoned test dataset
                rate = calc_accuracy(device, model, test_loader_poisoned)
                attack_success_rates.append(rate)

                tqdm.write(f"i={i}: accuracy {accuracy}, attack success rate {rate}")

    print("done training!")

    if should_save_stats:
        save_stats_plots(stats_file_name, accuracies, attack_success_rates)

    if should_save_model:
        save_model(model, model_file_name)


def defense(losses, optimizer, model, labels, is_poisoned=None, batch_idx=1):
    # Initialize a list to hold the gradients for each sample
    gradients = []

    # Backward pass for each sample
    for loss in losses:
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)

        # Save the gradients for each sample
        gradients.append(
            {name: param.grad.clone() for name, param in model.named_parameters()}
        )

    # save_gradient_means(gradients, labels, is_poisoned)
    # similarity = lambda grads, mean: torch.norm(grads-mean, dim=1)
    # aggregated_gradients, avg_weight_poisoned = aggregate_gradients_cosine(gradients, labels, is_poisoned, plot=(i==0))
    aggregated_gradients, avg_weight_poisoned = aggregate_all_params(
        gradients,
        labels,
        is_poisoned,
        plot=False,
        save_gradients=False,
        name_to_save=f"batch_{batch_idx}",
    )

    # Apply the aggregated gradients
    optimizer.zero_grad()
    for name, param in model.named_parameters():
        param.grad = aggregated_gradients[name]
    optimizer.step()
