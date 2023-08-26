import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    inplace_or_merge: str = settings.DEFAULT_INPLACE_OR_MERGE,
    batch_size: int = settings.DEFAULT_BATCH_SIZE,
    poison_rate: float = settings.DEFAULT_POISON_RATE,
):
    # Load the dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = dataset(root="data", train=True, download=True, transform=transform)
    test_data = dataset(root="data", train=False, download=True, transform=transform)

    train_loader_clean = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader_clean = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # add a small p% of poisoned samples to the train data
    train_loader_poisoned = DataLoader(
        PoisonedDataset(
            train_data,
            poison_rate,
            "all_to_target",
            target_class=settings.TARGET_CLASS,
            inplace_or_merge=inplace_or_merge,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    # poison all samples to test the attacks success rate
    test_loader_poisoned = DataLoader(
        PoisonedDataset(
            test_data,
            1.0,
            "all_to_target",
            target_class=settings.TARGET_CLASS,
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
    parser.add_argument(
        "--runtype", choices=["baseline", "attack", "defense"], help="Type of run"
    )
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "CIFAR10"],
        default="MNIST",
        help="Dataset to use (default: %(default)s)",
    )
    parser.add_argument(
        "--inplace_or_merge",
        choices=["inplace", "merge"],
        default=settings.DEFAULT_INPLACE_OR_MERGE,
        help="Inplace or merge operation (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=settings.DEFAULT_BATCH_SIZE,
        help="Batch size for training (default: %(default)d)",
    )
    parser.add_argument(
        "--poison_rate",
        type=float,
        default=settings.DEFAULT_POISON_RATE,
        help="Rate of poisoned samples in the dataset (default: %(default)f)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Save name for statistics (default: stats_{run_title}_accuracy_and_attack_success_rate)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.DEFAULT_TRAINING_EPOCHS,
        help="Number of epochs to run training (default: %(default)d)",
    )
    parser.add_argument(
        "--calc_every_n_iter",
        type=int,
        default=10,
        help="Save stats every given number of batches (default: %(default)d)",
    )
    parser.add_argument(
        "--calc-stats-on",
        choices=["test", "train"],
        default="train",
        help="Calculate stats (accuracy, attack success rate) on test/train dataset (default: %(default)s)",
    )
    args = parser.parse_args()

    run_title = "_".join(
        [
            args.runtype,
            args.dataset,
            f"poison_{args.poison_rate}_{args.inplace_or_merge}",
            f"epochs_{args.epochs}",
            f"batch_{args.batch_size}",
            f"calc_every_{args.calc_every_n_iter}_batches_on_{args.calc_stats_on}_dataset",
        ]
    )

    print(run_title)

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
    ) = get_data_loaders(
        dataset, args.inplace_or_merge, args.batch_size, args.poison_rate
    )

    # Initialize the network
    model = SimpleConvNet()

    if args.runtype == "baseline":
        train_loader = train_loader_clean
        print("training model on clean dataset, establishing model's baseline")
        assert (
            args.calc_stats_on == "test"
        ), "baseline stats can only be calculated on train dataset; run again with `--calc-stats-on test`"
    elif args.runtype == "attack":
        train_loader = train_loader_poisoned
        print("training model on poisoned dataset, establishing attack's baseline")
    elif args.runtype == "defense":
        train_loader = train_loader_poisoned
        print(
            "training model on poisoned dataset and defending against it, establishing defense's success"
        )

    train(
        device=device,
        model=model,
        epochs=args.epochs,
        defend=args.runtype == "defense",
        #
        train_loader=train_loader,
        test_loader_clean=test_loader_clean,
        test_loader_poisoned=test_loader_poisoned,
        #
        should_save_model=True,
        model_file_name=f"cnn_{run_title}",
        #
        should_save_stats=True,
        stats_file_name=args.save_name or f"stats_{run_title}",
        calc_stats_every_nth_iter=args.calc_every_n_iter,
        calc_stats_on_train_or_test=args.calc_stats_on,
    )


if __name__ == "__main__":
    main()
