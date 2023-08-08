import copy
from typing import Literal, Optional

import numpy as np
import torch


def white_square_watermark():
    # Define the watermark (backdoor trigger) and target class
    watermark = torch.zeros(1, 28, 28)  # A 28x28 black square
    # add a white square to the right bottom corner
    watermark[:, 25:27, 25:27] = 1
    return watermark


def gen_poisoned_samples(
    dataset: torch.utils.data.Dataset,
    poisoning_rate: float,
    attack_type: Literal["source_to_target", "all_to_target", "all_to_all_plus_one"],
    source_class: Optional[int] = None,
    target_class: Optional[int] = None,
    inplace_or_merge: Literal["inplace", "merge", "only_poisoned"] = "merge",
):
    assert 0 < poisoning_rate <= 1
    assert (
        inplace_or_merge == "inplace"
    ), "only `inplace` mode is supported for now, other modes are buggy"

    print(
        f"generating {int(len(dataset) * poisoning_rate)} poisoned samples from dataset of size {len(dataset)}, rate={poisoning_rate}, type={attack_type}, mode={inplace_or_merge}"
    )

    backdoor = white_square_watermark()

    indices_to_poison = np.random.choice(
        len(dataset),
        size=int(len(dataset) * poisoning_rate),
        replace=False,
    )

    if inplace_or_merge == "inplace":
        poisoned_data = copy.deepcopy(dataset)
    else:
        backdoored_data = []

    for idx in indices_to_poison:
        backdoored_image = torch.clip(dataset[idx][0] + backdoor, 0, 1)

        if attack_type == "all_to_target":
            backdoored_class = target_class
        else:
            # TODO:
            raise NotImplementedError()

        if inplace_or_merge == "inplace":
            poisoned_data.data[idx] = backdoored_image
            poisoned_data.targets[idx] = backdoored_class
        else:
            backdoored_data.append((backdoored_image, backdoored_class))

    if inplace_or_merge == "inplace":
        return poisoned_data
    elif inplace_or_merge == "merge":
        return torch.utils.data.dataset.ConcatDataset([dataset, backdoored_data])
    else:
        return backdoored_data


# class PoisonedDataset(torch.utils.data.Dataset):
#     def __init__(self, original_dataset, poison_rate, return_indices=True):
#         self.poisoned_data, self.poison_indices = poison_dataset(
#             original_dataset, poison_rate
#         )
#         self.return_indices = return_indices

#     def __getitem__(self, index):
#         data, target = self.poisoned_data[index]
#         is_poisoned = index in self.poison_indices
#         if self.return_indices:
#             return data, target, is_poisoned
#         else:
#             return data, target

#     def __len__(self):
#         return len(self.poisoned_data)
