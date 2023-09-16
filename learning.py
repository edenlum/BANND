from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aggregate_gradients import aggregate_all_params
from utils import green, save_model, save_stats_plots


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
    *,
    device,
    model,
    epochs: int = 1,
    defend: bool = False,
    similarity: str = "cosine",
    hard_threshold: Optional[float] = None,
    quantile_threshold: Optional[float] = None,
    #
    train_loader: DataLoader,
    test_loader_clean: Optional[DataLoader] = None,
    test_loader_poisoned: Optional[DataLoader] = None,
    #
    should_save_model: bool = False,
    model_file_name: Optional[str] = None,
    #
    should_save_stats: bool = False,
    stats_file_name: Optional[str] = None,
    plots_dir: str = "./plots/",
    calc_stats_every_nth_iter: int = 10,
    calc_stats_on_train_or_test: Literal["train", "test"] = "train",
):
    print("training model...")

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(**({"reduction": "none"} if defend else {}))

    accuracies = []
    attack_success_rates = []
    avg_weight_poisoned_list = []

    for _ in tqdm(range(epochs), desc="epoch"):
        for i, batch in enumerate(tqdm(train_loader, desc="batch")):
            if len(batch) == 2:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
            else:
                images, labels, is_poisoned = batch
                images, labels, is_poisoned = (
                    images.to(device),
                    labels.to(device),
                    is_poisoned.to(device),
                )

            # Forward pass
            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if defend:
                avg_weight_poisoned = defense(
                    loss,
                    optimizer,
                    model,
                    labels,
                    is_poisoned,
                    i,
                    similarity,
                    hard_threshold,
                    quantile_threshold,
                )
            else:
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if should_save_stats and (i % calc_stats_every_nth_iter == 0):
                if calc_stats_on_train_or_test == "train":
                    correct = outputs.argmax(dim=1) == labels
                    not_poisoned = correct[~is_poisoned]
                    poisoned = correct[is_poisoned]
                    accuracy = (not_poisoned.sum() / len(not_poisoned)).cpu()
                    rate = (poisoned.sum() / len(poisoned)).cpu()
                else:
                    # compute accuracy on clean test dataset
                    accuracy = calc_accuracy(device, model, test_loader_clean)
                    # compute attack success rate on poisoned test dataset
                    rate = calc_accuracy(device, model, test_loader_poisoned)

                accuracies.append(accuracy)
                attack_success_rates.append(rate)
                if defend:
                    avg_weight_poisoned_list.append(avg_weight_poisoned)
                    tqdm.write(
                        green(
                            f"i={i}: accuracy {accuracy}, attack success rate {rate}, avg_weight_poisoned {avg_weight_poisoned}"
                        )
                    )
                else:
                    tqdm.write(
                        green(f"i={i}: accuracy {accuracy}, attack success rate {rate}")
                    )

    print("done training!")

    if should_save_stats:
        save_stats_plots(
            stats_file_name,
            plots_dir,
            accuracies,
            attack_success_rates,
            calc_stats_every_nth_iter,
            avg_weight_poisoned_list if defend else None,
        )

    if should_save_model:
        save_model(model, model_file_name)


def defense(
    losses,
    optimizer,
    model,
    labels,
    is_poisoned=None,
    batch_idx=1,
    similarity="cosine",
    hard_threshold=None,
    quantile_threshold=None,
):
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

    if similarity == "l2":
        similarity = lambda grads, mean: torch.norm(grads - mean, dim=1)
    elif similarity == "cosine":
        similarity = F.cosine_similarity
    else:
        raise NotImplementedError
    # aggregated_gradients, avg_weight_poisoned = aggregate_gradients_cosine(gradients, labels, is_poisoned, plot=(i==0))
    aggregated_gradients, avg_weight_poisoned = aggregate_all_params(
        gradients,
        labels,
        is_poisoned,
        plot=False,
        # plot=(batch_idx == 0),
        save_gradients=False,
        name_to_save=f"batch_{batch_idx}",
        similarity=similarity,
        hard_threshold=hard_threshold,
        quantile_threshold=quantile_threshold,
    )

    # Apply the aggregated gradients
    optimizer.zero_grad()
    for name, param in model.named_parameters():
        param.grad = aggregated_gradients[name]
    optimizer.step()

    return avg_weight_poisoned
