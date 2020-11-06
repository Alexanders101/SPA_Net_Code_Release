import torch
from torch.utils.data import DataLoader

from ttbar.dataset import TTBarDataset
from ttbar.network.sherpa_module import SherpaModule
from ttbar.options import Options


class QuarkBaseNetwork(SherpaModule):
    def __init__(self, options: Options, client=None, trial=None):
        super().__init__(options, client, trial)

        self.hparams = self.options = options
        self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_datasets()

    @property
    def common_dataset_options(self):
        return {
            "max_jets": self.options.num_jets,
            "valid_subset": self.options.valid_subset,
            "event_mask": 2
        }

    @property
    def dataset(self):
        return TTBarDataset

    def create_datasets(self):
        training_dataset = self.dataset(self.options.event_file,
                                        index=self.options.train_test_split,
                                        **self.common_dataset_options)

        validation_dataset = self.dataset(self.options.event_file,
                                          index=self.options.train_test_split - 1,
                                          **self.common_dataset_options)

        testing_dataset = None
        if len(self.options.testing_file) > 0:
            testing_dataset = self.dataset(self.options.testing_file, **self.common_dataset_options)

        # Normalize all datasets but only use training statistics
        statistics = training_dataset.normalize()
        validation_dataset.normalize(*statistics)
        if testing_dataset is not None:
            testing_dataset.normalize(*statistics)

        return training_dataset, validation_dataset, testing_dataset

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = None

        if 'apex' in self.options.optimizer:
            try:
                import apex.optimizers

                if self.options.optimizer == 'apex_adam':
                    optimizer = apex.optimizers.FusedAdam

                elif self.options.optimizer == 'apex_lamb':
                    optimizer = apex.optimizers.FusedLAMB

                else:
                    optimizer = apex.optimizers.FusedSGD

            except ImportError:
                pass

        else:
            optimizer = getattr(torch.optim, self.options.optimizer)

        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.optimizer}.")
            print(f"Using pytorch Adam as a default.")
            optimizer = torch.optim.Adam

        return optimizer(self.parameters(), lr=self.options.learning_rate, weight_decay=self.options.l2_penalty)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_dataset,
                          batch_size=self.options.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.options.num_dataloader_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_dataset,
                          batch_size=self.options.batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=self.options.num_dataloader_workers,
                          pin_memory=True)

    def testing_dataloader(self) -> DataLoader:
        return DataLoader(self.testing_dataset,
                          batch_size=self.options.batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=self.options.num_dataloader_workers,
                          pin_memory=True)
