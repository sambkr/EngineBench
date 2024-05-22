import torch
from torch.utils.data import Dataset


class CustomConcatDataset(Dataset):
    """
    Concatenates several datasets without losing the attributes of each individual dataset.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_offsets = [0]  # Starting offset is 0

        for dataset in datasets[
            :-1
        ]:  # Exclude the last dataset from prefix sum computation
            self.dataset_offsets.append(self.dataset_offsets[-1] + len(dataset))

        self.metadata_ls = [d.metadata for d in datasets if hasattr(d, "metadata")]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        # If idx < offset, then idx must fall within the current dataset
        for dataset_idx, offset in enumerate(self.dataset_offsets[1:]):
            if idx < offset:
                break
        # Else, idx must fall within the last dataset
        else:
            dataset_idx = len(self.datasets) - 1

        # Adjust idx to be local to the dataset
        local_idx = idx - self.dataset_offsets[dataset_idx]

        return self.datasets[dataset_idx][local_idx]


def custom_collate_fn(batch):
    """
    Enables the mean and std values specific to each snapshot to propagate alongside
    the snapshots as dictionaries. Applies to the data loader class
    """
    tensor1s, tensor2s, tensor3s, dicts = zip(*batch)

    stacked_tensor1s = torch.stack(tensor1s)
    stacked_tensor2s = torch.stack(tensor2s)
    stacked_tensor3s = torch.stack(tensor3s)

    # Handle the dictionaries.
    list_of_dicts = [dict(d) for d in dicts]

    return stacked_tensor1s, stacked_tensor2s, stacked_tensor3s, list_of_dicts
