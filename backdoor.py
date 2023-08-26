import copy
from typing import Literal, Optional

import numpy as np
import torch


def white_square_watermark():
    # Define the watermark (backdoor trigger) and target class
    watermark = torch.zeros(1, 28, 28, dtype=torch.uint8)  # A 28x28 black square
    # add a white square to the right bottom corner
    watermark[:, 25:27, 25:27] = 255
    return watermark


def gen_poisoned_samples(
    dataset: torch.utils.data.Dataset,
    poisoning_rate: float,
    attack_type: Literal["source_to_target", "all_to_target", "all_to_all_plus_one"],
    source_class: Optional[int] = None,
    target_class: Optional[int] = None,
    inplace_or_merge: Literal["inplace", "merge"] = "merge",
):
    assert 0 < poisoning_rate <= 1
    num_poison = int(len(dataset) * poisoning_rate)
    print(
        f"generating {num_poison} poisoned samples from dataset of size {len(dataset)}, rate={poisoning_rate}, type={attack_type}, mode={inplace_or_merge}"
    )

    backdoor = white_square_watermark()

    indices_to_poison = np.random.choice(
        len(dataset),
        size=num_poison,
        replace=False,
    )

    poisoned_data = copy.deepcopy(dataset)
    backdoored_images = torch.clip(dataset.data[indices_to_poison] + backdoor, 0, 255)

    if attack_type == "all_to_target":
        backdoored_classes = torch.full(size=(num_poison,), fill_value=target_class)
    else:
        # TODO:
        raise NotImplementedError()

    if inplace_or_merge == "inplace":
        poisoned_data.data[indices_to_poison] = backdoored_images
        poisoned_data.targets[indices_to_poison] = backdoored_classes
        poison_indices = torch.zeros(len(poisoned_data.data), dtype=torch.bool)
        poison_indices[indices_to_poison] = True
    elif inplace_or_merge == "merge":
        poisoned_data.data = torch.concat(
            (poisoned_data.data, backdoored_images), dim=0
        )
        poisoned_data.targets = torch.concat(
            (poisoned_data.targets, backdoored_classes), dim=0
        )
        poison_indices = torch.zeros(len(poisoned_data.data), dtype=torch.bool)
        poison_indices[len(dataset) :] = True
    else:
        raise NotImplementedError()

    return poisoned_data, poison_indices


class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        poisoning_rate: float,
        attack_type: Literal[
            "source_to_target", "all_to_target", "all_to_all_plus_one"
        ],
        source_class: Optional[int] = None,
        target_class: Optional[int] = None,
        inplace_or_merge: Literal["inplace", "merge"] = "merge",
        *args,
        **kwargs,
    ):
        super(*args, **kwargs)
        self.poisoned_data, self.poison_indices = gen_poisoned_samples(
            dataset,
            poisoning_rate,
            attack_type,
            source_class,
            target_class,
            inplace_or_merge,
        )

    def __getitem__(self, index):
        data, target = self.poisoned_data[index]
        is_poisoned = self.poison_indices[index]
        return data, target, is_poisoned

    def __len__(self):
        return len(self.poisoned_data)
