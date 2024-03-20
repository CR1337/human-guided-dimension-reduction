import lightning as L
from torch.utils.data.dataloader import DataLoader
import datasets
import os
import numpy as np
from typing import List
import torch


class DataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.train_file = str(self.data_dir / args.train_file)
        self.val_file = str(self.data_dir / args.val_file)
        self.test_file = str(self.data_dir / args.test_file)
        self.max_landmarks = args.max_landmarks
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size

    def prepare_data(self) -> None:
        # Prepare data by checking cache and processing datasets if necessary
        cache_exists, cache_path = self._get_dataset_cache_path()
        if not cache_exists:
            print(
                f"Could not find cached processed dataset: {cache_path}, "
                "creating it now..."
            )
            # Load and process dataset if not cached
            processed_datasets = self.load_and_process_dataset()
            print(f"Saving dataset to {cache_path}...")
            processed_datasets.save_to_disk(cache_path)
        else:
            print(f"Found cached processed dataset: {cache_path}.")

    def setup(self, stage):
        # Setup datasets for training or validation stage
        cache_exists, cache_path = self._get_dataset_cache_path()
        assert (
            cache_exists
        ), (
            f"Could not find cached processed dataset: {cache_path}, "
            "should have been created in prepare_data()"
        )

        print(f"Loading cached processed dataset from {cache_path}...")
        processed_datasets = datasets.load_from_disk(cache_path)

        # Assign datasets and data collator for training and validation
        self.train_dataset = processed_datasets["train"]
        self.val_dataset = processed_datasets["val"]
        self.test_dataset = processed_datasets["test"]

    def load_and_process_dataset(self):
        # Define paths for training and validation data files
        data_files = {
            "train": self.train_file,
            "val": self.val_file,
            "test": self.test_file,
        }

        print("Loading raw dataset...")

        # Create temporary directory for dataset caching,
        # if disk space conservation is enabled
        train_val_datasets = datasets.load_dataset(
            "json",
            data_files=data_files,
            name=str(self.data_dir).replace("/", "_"),
            num_proc=1,
        )

        processed_datasets = self.process_datasets(train_val_datasets)

        print(f"Finished processing datasets: {processed_datasets}")

        return processed_datasets

    def process_datasets(self, train_val_datasets):
        """
        We want to pad the inputs and convert to tensors
        """
        processed_datasets = train_val_datasets.map(
            make_process_function(self.max_landmarks),
            batched=True,
            num_proc=1,
            desc="Running preprocessing on every input in dataset",
        )

        return processed_datasets

    def _get_dataset_cache_path(self):
        process_function = make_process_function(self.max_landmarks)
        process_fn_hash = datasets.fingerprint.Hasher.hash(process_function)

        # Define the directory and file path for the cached data
        processed_data_dir = str(self.data_dir / "processed")
        cache_path = os.path.join(
            processed_data_dir,
            f"seq_len_{self.max_landmarks}process_fn_hash"
            f"{process_fn_hash}.arrow",
        )

        # Determine if a valid cache file exists and return its path
        if os.path.exists(cache_path):
            return True, cache_path
        return False, cache_path

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=self.num_workers)

    def collate_fn(self, examples):
        inputs = torch.stack([torch.Tensor(example["input"]) for example in examples])
        labels = torch.stack([torch.Tensor(example["label"]) for example in examples])
        masks = torch.stack([torch.Tensor(example["mask"]).int() for example in examples])
        return {
            "input": inputs,
            "label": labels,
            "mask": masks
        }


def make_process_function(max_landmarks):
    def process_function(examples):
        inputs = [
            process_single_input(np.asarray(inp), max_landmarks)
            for inp in examples["input"]
        ]
        labels = [
            process_single_input(np.asarray(label), max_landmarks)
            for label in examples["label"]
        ]
        masks = [
            torch.where(label != -1)[0] for label in labels
        ]

        # Pad masks to label length
        masks = [
            torch.cat(
                [
                    mask,
                    torch.ones(len(labels[0]) - len(mask)) * -1,
                ]
            )
            for mask in masks
        ]

        return {
            "input": inputs,
            "label": labels,
            "mask": masks,
        }

    return process_function

def process_single_input(inp, max_landmarks):
    def _pad_array(array):
        return np.pad(
            array,
            ((0, max_landmarks - array.shape[0]),
             (0, max_landmarks - array.shape[1])),
            mode="constant",
            constant_values=-1,
        )

    def _take_upper_triangle(array: np.ndarray) -> List[float]:
        indices = np.triu_indices(max_landmarks, k=1)
        values = array[indices]
        return values.tolist()

    return torch.tensor(_take_upper_triangle(_pad_array(inp)))
