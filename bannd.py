import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

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

def get_data_loaders(
    dataset: datasets.VisionDataset,
    inplace_or_merge: str = settings.INPLACE_OR_MERGE,
    batch_size: int = settings.BATCH_SIZE,
    poison_rate: float = settings.POISON_RATE,
):
    # Load the dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = dataset(
        root="data", train=True, download=True, transform=transform
    )
    test_data = dataset(
        root="data", train=False, download=True, transform=transform
    )

    train_loader_clean = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader_clean = DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )

    # add a small p% of poisoned samples to the train data
    train_loader_poisoned = DataLoader(
        gen_poisoned_samples(
            train_data,
            poison_rate,
            "all_to_target",
            target_class=1,
            inplace_or_merge=inplace_or_merge,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    # poison all samples to test the attacks success rate
    test_loader_poisoned = DataLoader(
        gen_poisoned_samples(
            test_data,
            1.0,
            "all_to_target",
            target_class=1,
            inplace_or_merge="inplace",
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return (
        train_loader_clean,
        train_loader_poisoned,
        test_loader_clean,
        test_loader_poisoned,
    )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network.")
    parser.add_argument('--runtype', choices=["baseline", "attack", "defend"], help='Type of run')
    parser.add_argument('--dataset', choices=["MNIST", "CIFAR10"], help='Dataset to use')
    parser.add_argument('--inplace_or_merge', choices=["inplace", "merge"], help='Inplace or merge operation')
    parser.add_argument('--batch_size', type=int, default=settings.BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--poison_rate', type=float, default=settings.POISON_RATE, help='Rate of poisoned samples in the dataset')
    parser.add_argument('--save_name', type=str, default=None, help='Save name for statistics')

    args = parser.parse_args()
    
    if args.save_name is None:
        args.save_name = f"stats_{args.runtype}_accuracy_and_attack_success_rate"

    if args.dataset == "MNIST":
        dataset = datasets.MNIST
    elif args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10
    else:
        raise NotImplementedError()
      
    (
        train_loader_clean,
        train_loader_poisoned,
        test_loader_clean,
        test_loader_poisoned,
    ) = get_data_loaders(dataset, args.inplace_or_merge, args.batch_size, args.poison_rate)

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
            stats_file_name=args.save_name,
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
            stats_file_name=args.save_name,
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
    # main("baseline")
    main()
