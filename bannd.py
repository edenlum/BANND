import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import settings
from aggregate_gradients import *
from nets import *
from poison_dataset import *
from utils import *

torch.manual_seed(settings.SEED)
random.seed(settings.SEED)
np.random.seed(settings.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_normal(name, model, train_loader, test_loader_clean, test_loader_poisoned):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    print("Training the model normally...")
    # Training loop
    accuracies = []
    attack_success_rates = []

    for epoch in range(1):
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            model.train()
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    # Compute accuracy on clean test set
                    accuracy = test(model, test_loader_clean)
                    accuracies.append(accuracy)

                    # Compute attack success rate on poisoned test set
                    attack_success_rate = test(model, test_loader_poisoned)
                    attack_success_rates.append(attack_success_rate)

    print("Training is complete!")
    plot_through_training("training_stats_no_def", accuracies, attack_success_rates)

    # save the model
    print("Saving the model...")
    torch.save(model.state_dict(), f"./data/models/{name}.pth")


def test(model, test_loader):
    # Test the model
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print("Accuracy: {}%".format(accuracy))
    return accuracy


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
        for i, (images, labels, is_poisoned) in enumerate(tqdm.tqdm(train_loader)):
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
                    accuracy = test(model, test_loader_clean)
                    accuracies.append(accuracy)

                    # Compute attack success rate on poisoned test set
                    attack_success_rate = test(model, test_loader_poisoned)
                    attack_success_rates.append(attack_success_rate)

    print("Training is complete!")
    plot_through_training(
        "training_stats_with_def", accuracies, attack_success_rates, avg_weight_ratios
    )

    # save the model
    print("Saving the model...")
    torch.save(model.state_dict(), f"./data/models/{model_name}.pth")
    print(f"Model saved to ./data/models/{model_name}.pth")


def main():
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # Initialize the network and optimizer
    model = SimpleConvNet()

    # print("Training normal CNN on normal dataset")
    # train_normal("mnist_cnn", model, train_loader)
    # test(model, test_loader)

    # Create a DataLoader from the poisoned dataset
    watermark = white_square_watermark()
    poisoned_train_loader = DataLoader(
        PoisonedDataset(train_data, watermark, 0.01), batch_size=256
    )
    poisoned_test_loader = DataLoader(
        PoisonedDataset(test_data, watermark, 1.0), batch_size=256
    )

    # Train the model with the poisoned dataset
    print("Training normally on backdoored dataset")
    train_normal(
        "mnist_cnn_backdoor",
        model,
        poisoned_train_loader,
        test_loader,
        poisoned_test_loader,
    )
    # # model.load_state_dict(torch.load("./data/models/mnist_cnn_backdoor.pth"))
    # print("Testing the model with a backdoor on the clean test set")
    # test(model, test_loader)
    # print("Testing the model with a backdoor on the poisoned test set")
    # test(model, poisoned_test_loader)

    print("Training with defense on backdoored dataset")
    defended_model = SimpleConvNet()

    # train_defense("mnist_cnn_backdoor_defense", defended_model, poisoned_train_loader, test_loader, poisoned_test_loader)
    # defended_model.load_state_dict(torch.load("./data/models/mnist_cnn_backdoor_defense.pth"))
    # print("Testing the model with a backdoor on the clean test set")
    # test(defended_model, test_loader)
    # print("Testing the model with a backdoor on the poisoned test set")
    # test(defended_model, poisoned_test_loader)


if __name__ == "__main__":
    main()
