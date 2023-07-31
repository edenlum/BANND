import torch
import copy
import numpy as np


def white_square_watermark():
  # Define the watermark (backdoor trigger) and target class
  watermark = torch.zeros(1, 28, 28)  # A 28x28 black square
  # add a white square to the right bottom corner
  watermark[:, 25:27, 25:27] = 1.0
  return watermark

def poison_dataset(data, watermark, poison_rate=0.05, target_class=1):
    num_poison = int(len(data) * poison_rate)
    poison_indices = np.random.choice(len(data), size=num_poison, replace=False)
    poisoned_data = copy.deepcopy(data)
    for i in poison_indices:
        poisoned_data.data[i] = torch.clip(poisoned_data.data[i] + watermark, 0, 1)
        poisoned_data.targets[i] = target_class
    return poisoned_data, poison_indices


class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, watermark, poison_rate, return_indices=True):
        self.poisoned_data, self.poison_indices = poison_dataset(original_dataset,
                                                                 watermark=watermark,
                                                                 poison_rate=poison_rate)

    def __getitem__(self, index):
        data, target = self.poisoned_data[index]
        is_poisoned = index in self.poison_indices
        if self.return_indices:
            return data, target, is_poisoned
        else:
            return data, target

    def __len__(self):
        return len(self.poisoned_data)