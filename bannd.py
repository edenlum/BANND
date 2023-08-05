import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import settings
from aggregate_gradients import *
from backdoor import *
from learning import *
from nets import *
from utils import *

torch.manual_seed(settings.SEED)
random.seed(settings.SEED)
np.random.seed(settings.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_defense(
    model_name, model, train_loader, test_loader_clean, test_loader_poisoned, epochs=1
):
    print("Training the model with a backdoor and a defense...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # loss_fn = nn.CrossEntropyLoss(reduction='none')
    model.to(device)

    num_classes = 10  # Number of classes in your dataset
    smoothing = 0.1  # Label smoothing factor

    loss_fn = nn.KLDivLoss(reduction="none")

    accuracies = []
    attack_success_rates = []
    avg_weight_ratios = []

    for epoch in range(epochs):
        for i, (images, labels, is_poisoned) in enumerate(tqdm(train_loader)):
            model.train()
            images, labels = images.to(device), labels.to(device)
            log_outputs = torch.log_softmax(model(images), dim=1)

            # Apply label smoothing to the target labels
            smoothed_labels = smooth_labels(labels, num_classes, smoothing).to(device)
            # Compute the loss using the smoothed labels
            losses = loss_fn(log_outputs, smoothed_labels).sum(dim=1)
            # Initialize a list to hold the gradients for each sample
            gradients = []

            # Backward pass for each sample
            for loss in losses:
                optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)

                # Save the gradients for each sample
                gradients.append(
                    {
                        name: param.grad.clone()
                        for name, param in model.named_parameters()
                    }
                )

            # save_gradient_means(gradients, labels, is_poisoned)
            # similarity = lambda grads, mean: torch.norm(grads-mean, dim=1)
            # aggregated_gradients, avg_weight_poisoned = aggregate_gradients_cosine(gradients, labels, is_poisoned, plot=(i==0))
            aggregated_gradients, avg_weight_poisoned = aggregate_all_params(
                gradients,
                labels,
                is_poisoned,
                plot=(i == 0),
                save_gradients=False,
                name_to_save=f"batch_{i}",
            )
            avg_weight_ratios.append(avg_weight_poisoned)

            # Apply the aggregated gradients
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                param.grad = aggregated_gradients[name]
            optimizer.step()

            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    # Compute accuracy on clean test set
                    accuracy = calc_accuracy(device, model, test_loader_clean)
                    accuracies.append(accuracy)

                    # Compute attack success rate on poisoned test set
                    attack_success_rate = calc_accuracy(
                        device, model, test_loader_poisoned
                    )
                    attack_success_rates.append(attack_success_rate)

    print("Training is complete!")
    save_stats_plots(
        "training_stats_with_def", accuracies, attack_success_rates, avg_weight_ratios
    )

    # save the model
    print("Saving the model...")
    torch.save(model.state_dict(), f"./data/models/{model_name}.pth")
    print(f"Model saved to ./data/models/{model_name}.pth")


def get_data_loaders():
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader_clean = DataLoader(
        train_data, batch_size=settings.BATCH_SIZE, shuffle=True
    )
    test_loader_clean = DataLoader(
        test_data, batch_size=settings.BATCH_SIZE, shuffle=True
    )

    # add a small p% of poisoned samples to the train data
    train_loader_poisoned = DataLoader(
        torch.utils.data.dataset.ConcatDataset(
            [
                train_data,
                gen_poisoned_samples(
                    train_data,
                    settings.POISON_RATE,
                    "all_to_target",
                    target_class=settings.TARGET_CLASS,
                ),
            ]
        ),
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
    )
    # poison all samples to test the attacks success rate
    test_loader_poisoned = DataLoader(
        gen_poisoned_samples(
            test_data,
            1.0,
            "all_to_target",
            target_class=settings.TARGET_CLASS,
        ),
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
    )

    return (
        train_loader_clean,
        train_loader_poisoned,
        test_loader_clean,
        test_loader_poisoned,
    )


def main(run_type):
    (
        train_loader_clean,
        train_loader_poisoned,
        test_loader_clean,
        test_loader_poisoned,
    ) = get_data_loaders()

    # Initialize the network
    model = SimpleConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if run_type == "baseline":
        print("training model on clean dataset, establishing model's baseline")
        train(
            device,
            model,
            optimizer,
            criterion,
            train_loader_clean,
            should_save_model=True,
            model_file_name="cnn_baseline",
            should_save_stats=True,
            stats_file_name="stats_baseline_accuracy_and_attack_success_rate",
            test_loader_clean=test_loader_clean,
            test_loader_poisoned=test_loader_poisoned,
            calc_states_every_nth_iter=10,
        )

    elif run_type == "attack":
        print("training model on poisoned dataset, establishing attack's baseline")
        train(
            device,
            model,
            optimizer,
            criterion,
            train_loader_poisoned,
            should_save_model=True,
            model_file_name="cnn_after_attack",
            should_save_stats=True,
            stats_file_name="stats_attack_accuracy_and_success",
            test_loader_clean=test_loader_clean,
            test_loader_poisoned=test_loader_poisoned,
            calc_states_every_nth_iter=10,
        )

    elif run_type == "defend":
        raise NotImplementedError()
    # # model.load_state_dict(torch.load("./data/models/mnist_cnn_backdoor.pth"))
    # print("Testing the model with a backdoor on the clean test set")
    # calc_accuracy(model, test_loader)
    # print("Testing the model with a backdoor on the poisoned test set")
    # calc_accuracy(model, poisoned_test_loader)

    # print("Training with defense on backdoored dataset")
    # defended_model = SimpleConvNet()

    # train_defense("mnist_cnn_backdoor_defense", defended_model, poisoned_train_loader, test_loader, poisoned_test_loader)
    # defended_model.load_state_dict(torch.load("./data/models/mnist_cnn_backdoor_defense.pth"))
    # print("Testing the model with a backdoor on the clean test set")
    # calc_accuracy(defended_model, test_loader)
    # print("Testing the model with a backdoor on the poisoned test set")
    # calc_accuracy(defended_model, poisoned_test_loader)


if __name__ == "__main__":
    main("baseline")
    # main("attack")
