import gzip
import os
import pickle
from textwrap import wrap

import numpy as np
import torch
from matplotlib import pyplot as plt


def save_gradients_labels_poisoned(
    name, gradients, labels, is_poisoned, save_dir="gradients"
):
    os.makedirs(save_dir, exist_ok=True)
    with gzip.open(os.path.join(save_dir, f"{name}.pkl.gz"), "wb") as f:
        pickle.dump((gradients, labels, is_poisoned), f)


def load_gradients_labels_poisoned(name, save_dir="gradients"):
    with gzip.open(os.path.join(save_dir, f"{name}.pkl.gz"), "rb") as f:
        gradients, labels, is_poisoned = pickle.load(f)
    return gradients, labels, is_poisoned


def save_gradient_means(gradients, labels, is_poisoned, save_dir="gradient_means"):
    gradients = {
        name: [sample[name] for sample in gradients] for name in gradients[0].keys()
    }

    # Get the unique labels (classes)
    unique_labels = torch.unique(labels)

    labels = list(labels.cpu().numpy())
    is_poisoned = list(is_poisoned.cpu().numpy())

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize a dictionary to hold the mean gradients for each group
    mean_gradients = {label.item(): {} for label in unique_labels}
    mean_gradients["poisoned"] = {}

    # Divide the gradients into groups and compute the mean for each group
    for name, grad_list in gradients.items():
        # Divide the gradients into groups
        grad_groups = {label.item(): [] for label in unique_labels}
        grad_groups["poisoned"] = []
        for grad, label, poisoned in zip(grad_list, labels, is_poisoned):
            if poisoned:
                grad_groups["poisoned"].append(grad)
            else:
                grad_groups[label.item()].append(grad)

        # Compute the mean gradient for each group
        for group, grads in grad_groups.items():
            if grads:
                mean_gradients[group][name] = torch.stack(grads).mean(dim=0)
            else:
                mean_gradients[group][name] = None

    # Save the mean gradients to disk
    for group, grads in mean_gradients.items():
        torch.save(grads, os.path.join(save_dir, f"mean_gradients_{group}.pt"))


def smooth_labels(labels, num_classes, smoothing=0.1):
    # Convert labels to one-hot vectors
    one_hot_labels = torch.zeros(labels.size(0), num_classes).to(labels.device)
    one_hot_labels.scatter_(1, labels.unsqueeze(-1), 1)

    # Apply label smoothing
    smooth_labels = one_hot_labels * (1 - smoothing) + smoothing / num_classes
    return smooth_labels


def save_stats_plots(
    file_name,
    plots_dir,
    accuracies,
    attack_success_rates,
    every_x_batches,
    avg_weight_ratios=None,
):
    num_plots = 2 if avg_weight_ratios is None else 3

    # can just use `str.removeprefix()` in python > 3.10
    title = file_name
    if title.startswith("stats_"):
        title = title[len("stats_") :]

    plt.figure(figsize=(num_plots * 5, 5))
    plt.suptitle("\n".join(wrap(title)))

    plt.subplot(1, num_plots, 1)
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.xlabel(f"Batch/{every_x_batches}")
    plt.ylabel("Accuracy")
    plt.ylim(0, 100)
    plt.yticks(list(range(0, 106, 5)))

    plt.subplot(1, num_plots, 2)
    plt.plot(attack_success_rates)
    plt.title("Attack Success Rate")
    plt.xlabel(f"Batch/{every_x_batches}")
    plt.ylabel("Success Rate")
    plt.ylim(0, 100)
    plt.yticks(list(range(0, 106, 5)))

    if avg_weight_ratios is not None:
        plt.subplot(1, num_plots, 3)
        plt.plot(avg_weight_ratios)
        plt.title("Average Weight Ratio")
        plt.xlabel("Batch")
        plt.ylabel("Ratio")

    plt.subplots_adjust(top=0.8)

    path = os.path.join(plots_dir, f"{file_name}.png")
    print(blue(f"saving stats (model accuracies and attack success rates) to {path}"))
    plt.savefig(path)

    plt.close()


def plot_sim_scores(file_name, grads, is_poisoned, class_title):
    plt.title(f"AVG Gradient Similarity Score for class {class_title}")
    grads_indices = np.arange(grads.size(0))
    print(grads_indices)
    plt.scatter(grads_indices, grads, c=is_poisoned)
    path = f"plots/{file_name}.png"
    plt.savefig(path)


def save_model(model, file_name):
    path = f"data/models/{file_name}.pth"
    print(blue(f"saving model to {path}"))
    torch.save(model.state_dict(), path)


def red(s: str) -> str:
    return f"\033[0;31m{s}\033[0m"


def green(s: str) -> str:
    return f"\033[0;32m{s}\033[0m"


def blue(s: str) -> str:
    return f"\033[0;34m{s}\033[0m"


def bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"
