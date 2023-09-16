import argparse
import random
from typing import Literal, Optional

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


def bannd(
    runtype: Literal["baseline", "attack", "defense"],
    dataset: Literal["MNIST", "CIFAR10"] = "MNIST",
    # inplace_or_merge=settings.DEFAULT_INPLACE_OR_MERGE,
    # batch_size=settings.DEFAULT_BATCH_SIZE,
    poison_rate: float = settings.DEFAULT_POISON_RATE,
    # save_name=None,
    # epochs=settings.DEFAULT_TRAINING_EPOCHS,
    # calc_every_n_iter=10,
    # calc_stats_on="test",
    # similarity="cosine",
    hard_threshold: Optional[float] = None,
    quantile_threshold: Optional[float] = None,
    plots_dir: str = "./plots/",
):
    if runtype == "defense":
        assert (hard_threshold is None and quantile_threshold is not None) or (
            hard_threshold is not None and quantile_threshold is None
        ), "pass either --hard-threshold or --quantile-threshold, not both!"

    run_title = "_".join(
        [
            runtype,
            dataset,
            *([f"p{poison_rate}"] if runtype in ["attack", "defense"] else []),
            *(
                [
                    f"ht{hard_threshold}"
                    if hard_threshold is not None
                    else f"qt{quantile_threshold}"
                ]
                if runtype == "defense"
                else []
            ),
            # f"p{poison_rate}-{inplace_or_merge}",
            # f"e{epochs}",
            # f"b{batch_size}",
            # f"s-{similarity}",
            # f"every-{calc_every_n_iter}-on-{calc_stats_on}",
        ]
    )

    print(bold(f"title: {run_title}"))

    inplace_or_merge = settings.DEFAULT_INPLACE_OR_MERGE
    batch_size = settings.DEFAULT_BATCH_SIZE
    epochs = settings.DEFAULT_TRAINING_EPOCHS
    calc_every_n_iter = 10
    calc_stats_on = "test"
    similarity = "cosine"

    print(
        bold(
            f"using default configurations (not part of title, for brevity): p-{inplace_or_merge}_e{epochs}_b{batch_size}_s-{similarity}_every-{calc_every_n_iter}-on-{calc_stats_on}"
        )
    )

    if dataset == "MNIST":
        dataset = datasets.MNIST
    elif dataset == "CIFAR10":
        dataset = datasets.CIFAR10
    else:
        raise NotImplementedError()

    (
        train_loader_clean,
        train_loader_poisoned,
        test_loader_clean,
        test_loader_poisoned,
    ) = get_data_loaders(dataset, inplace_or_merge, batch_size, poison_rate)

    # Initialize the network
    model = SimpleConvNet()

    if runtype == "baseline":
        train_loader = train_loader_clean
        print("training model on clean dataset, establishing model's baseline")
        assert (
            calc_stats_on == "test"
        ), "baseline stats can only be calculated on train dataset; run again with `--calc-stats-on test`"
    elif runtype == "attack":
        train_loader = train_loader_poisoned
        print("training model on poisoned dataset, establishing attack's baseline")
    elif runtype == "defense":
        train_loader = train_loader_poisoned
        print(
            "training model on poisoned dataset and defending against it, establishing defense's success"
        )

    train(
        device=device,
        model=model,
        epochs=epochs,
        defend=runtype == "defense",
        similarity=similarity,
        hard_threshold=hard_threshold,
        quantile_threshold=quantile_threshold,
        #
        train_loader=train_loader,
        test_loader_clean=test_loader_clean,
        test_loader_poisoned=test_loader_poisoned,
        #
        should_save_model=True,
        model_file_name=f"cnn_{run_title}",
        #
        should_save_stats=True,
        stats_file_name=f"stats_{run_title}",
        plots_dir=plots_dir,
        calc_stats_every_nth_iter=calc_every_n_iter,
        calc_stats_on_train_or_test=calc_stats_on,
    )


def main():
    parser = argparse.ArgumentParser(description="Train, attack, and defend a CNN.")
    parser.add_argument(
        "--runtype", choices=["baseline", "attack", "defense"], help="Type of run"
    )
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "CIFAR10"],
        default="MNIST",
        help="Dataset to use (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--inplace_or_merge",
    #     choices=["inplace", "merge"],
    #     default=settings.DEFAULT_INPLACE_OR_MERGE,
    #     help="Inplace or merge operation (default: %(default)s)",
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=settings.DEFAULT_BATCH_SIZE,
    #     help="Batch size for training (default: %(default)d)",
    # )
    parser.add_argument(
        "--poison_rate",
        type=float,
        default=settings.DEFAULT_POISON_RATE,
        help="Rate of poisoned samples in the dataset (default: %(default)f)",
    )
    # parser.add_argument(
    #     "--save_name",
    #     type=str,
    #     default=None,
    #     help="Save name for statistics (default: stats_{run_title}_accuracy_and_attack_success_rate)",
    # )
    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=settings.DEFAULT_TRAINING_EPOCHS,
    #     help="Number of epochs to run training (default: %(default)d)",
    # )
    # parser.add_argument(
    #     "--calc_every_n_iter",
    #     type=int,
    #     default=10,
    #     help="Save stats every given number of batches (default: %(default)d)",
    # )
    # parser.add_argument(
    #     "--calc-stats-on",
    #     choices=["test", "train"],
    #     default="test",
    #     help="Calculate stats (accuracy, attack success rate) on test/train dataset (default: %(default)s)",
    # )
    # parser.add_argument(
    #     "--similarity",
    #     choices=["cosine", "l2"],
    #     default="cosine",
    #     help="Choose the similarity function (default: %(default)s)",
    # )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        help="Threshold to discard samples that have less than that in their similarity score. `softmax()` is applied to the samples according to their similarity score after applying the threshold. Pass `0` for no threshold, i.e., use all samples.",
    )
    parser.add_argument(
        "--quantile-threshold",
        type=float,
        help="Quantile Threshold to discard samples that have less than that in their similarity score. `softmax()` is applied to the samples according to their similarity score after applying the threshold. Pass `0` for no threshold, i.e., use all samples.",
    )

    parser.add_argument(
        "--plots-dir",
        type=str,
        default="./plots/",
        help="Output directory of plots (hint: `from google.colab import drive; drive.mount('/content/drive')` and then save the plots to drive in case the colab session disconnects) (default: %(default)s)",
    )
    args = parser.parse_args()

    bannd(
        args.runtype,
        args.dataset,
        # args.inplace_or_merge,
        # args.batch_size,
        args.poison_rate,
        # args.save_name,
        # args.epochs,
        # args.calc_every_n_iter,
        # args.calc_stats_on,
        # args.similarity,
        args.hard_threshold,
        args.quantile_threshold,
        args.plots_dir,
    )


if __name__ == "__main__":
    main()
