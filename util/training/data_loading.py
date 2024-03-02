import lightning as L
from torch.utils.data.dataloader import DataLoader
import datasets
from print_on_steroids import logger

class DataModule(L.LightningDataModule):
    def __init__(self, args, max_input_size: int):
        super().__init__()
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.train_file = args.train_file
        self.val_file = args.val_file
        self.test_file = args.test_file
        self.max_input_size = max_input_size

    def prepare_data(self) -> None:
        # Prepare data by checking cache and processing datasets if necessary
        cache_exists, cache_path = self._get_dataset_cache_path(self.tokenizer_path)
        if not cache_exists:
            logger.info(f"Could not find cached processed dataset: {cache_path}, creating it now...")
            # Load and process dataset if not cached
            processed_datasets = self.load_and_process_dataset(self.tokenizer, str(self.data_dir / "tokenized"))
            logger.info(f"Saving dataset to {cache_path}...")
            processed_datasets.save_to_disk(cache_path, num_proc=self.args.preprocessing_workers)
        else:
            logger.success(f"Found cached processed dataset: {cache_path}.")

    def setup(self, stage):
        # Setup datasets for training or validation stage
        cache_exists, cache_path = self._get_dataset_cache_path(self.tokenizer_path)
        assert (
            cache_exists
        ), f"Could not find cached processed dataset: {cache_path}, should have been created in prepare_data()"

        logger.info(f"Loading cached processed dataset from {cache_path}...", rank0_only=False)
        processed_datasets = datasets.load_from_disk(cache_path)

        # Assign datasets and data collator for training and validation
        self.train_dataset = processed_datasets["train"]
        self.val_dataset = processed_datasets["val"]
        self.test_dataset = processed_datasets["test"]

    def load_and_process_dataset(self, tokenizer, tokenized_data_dir):
        # Determine the file format of the dataset (txt, jsonl, etc.)
        extension = "csv"

        # Define paths for training and validation data files
        data_files = {"train": self.train_file, "val": self.val_file, "test": self.test_file}

        logger.info("Loading raw dataset...")

        # Create temporary directory for dataset caching, if disk space conservation is enabled
        train_val_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            name=str(self.data_dir).replace("/", "_"),
            num_proc=1,
        )

        # Debug logging for the first two samples of the training dataset
        if self.local_rank == 0:
            logger.debug((train_val_datasets, train_val_datasets["train"][:2]))

        processed_datasets = self.process_datasets(tokenizer, train_val_datasets, tokenized_data_dir)

        logger.success(
            f"Rank {self.local_rank} | Finished processing datasets: {processed_datasets} | First sample len: {len(processed_datasets['train'][0]['input_ids'])}"
        )

        return processed_datasets

    def process_datasets(self, tokenizer, train_val_datasets, tokenized_data_dir):
        """
        We want to pad the inputs and convert to tensors
        """
        return 1

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)