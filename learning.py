import torch
from tqdm import tqdm

import settings
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
    optimizer,
    criterion,
    train_loader,
    should_save_model=False,
    model_file_name=None,
    should_save_stats=False,
    stats_file_name=None,
    test_loader_clean=None,
    test_loader_poisoned=None,
    calc_states_every_nth_iter=10,
):
    print("training model...")

    model.to(device)

    accuracies = []
    attack_success_rates = []

    for _ in tqdm(range(settings.TRAINING_EPOCHS), desc="epoch"):
        for i, batch in enumerate(tqdm(train_loader, desc="batch")):
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)

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
