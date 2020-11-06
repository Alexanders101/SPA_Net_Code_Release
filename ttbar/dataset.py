from typing import Optional, Union, Tuple, List, Callable
from collections import OrderedDict
from pathlib import Path

import numpy as np

from tqdm import tqdm
from h5py import File as H5File

import torch
from torch.utils.data import Dataset


def dict_zip(*dicts):
    """ Generalization of the `zip` function to dictionaries """
    for i in set(dicts[0]).intersection(*dicts[1:]):
        yield (i,) + tuple(d[i] for d in dicts)


class TTBarDataset(Dataset):
    def __init__(self,
                 hdf5_path: str,
                 index: Union[Tuple[float, float], float] = 1.0,
                 max_jets: Optional[int] = None,
                 event_mask: Optional[int] = 2,
                 valid_subset: bool = False,
                 use_cache: bool = True,
                 sparse_format: bool = False):
        """ Create a dataset describing particle collision jets.

        Options
        ----------
        hdf5_path: str
            Links to base hdf5 file containing the data.
        index: (float, float) or float
            Defines the subset to take of the data. This is useful for creating train-test splits.
        max_jets: int, optional
            The maximum number of jets that can be present during an event.
        event_mask: int, optional
            Limit the events to those which have the given number of top quarks.
        use_cache: bool
            If true, then this class will create and read from any present cache file for a given hdf5 file.
        """

        jet_suffix = f'{max_jets}_jets.' if max_jets is not None else 'all_jets.'
        mask_suffix = f'{event_mask}_top.' if event_mask is not None else ''
        valid_suffix = 'valid.' if valid_subset else ''
        sparse_suffix = 'sparse.' if sparse_format else ''
        cache_file = Path(f"{hdf5_path}.{jet_suffix}{mask_suffix}{valid_suffix}{sparse_suffix}{self.suffix}cache")

        if use_cache and cache_file.exists():
            extra_message = 'Simplified to 6 valid jets.' if valid_subset else ''
            print(f"Loading from cache file for: {hdf5_path} with {max_jets} Jets. {extra_message}")
            self._load_from_cache(cache_file)

        else:
            self._load_data(hdf5_path, max_jets, event_mask, valid_subset, sparse_format)
            self._save_to_cache(cache_file)

        if valid_subset:
            self._limit_data_to_subset()

        if not isinstance(index, (list, tuple)):
            index = (0.0, index) if index > 0 else (1.0 + index, 1.0)

        # Take subsection of the data
        index = (int(round(index[0] * self.num_samples)), int(round(index[1] * self.num_samples)))
        indices = np.arange(self.num_samples)[index[0]:index[1]]
        self.num_samples = indices.shape[0]

        print(index)

        # Create data objects
        self.mask = self.mask[indices]
        self.source = self.source[indices]
        self.indices = indices

        if sparse_format:
            self.targets = OrderedDict({key: self.contiguous_subset_sparse(val, indices)
                                        for key, val in self.targets.items()})
        else:
            self.targets = OrderedDict({key: val[indices] for key, val in self.targets.items()})

        self.sparse_format = sparse_format

        # Create transformation objects
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    @property
    def suffix(self):
        return ""

    # =============================================================================================
    # Initialization Methods
    # =============================================================================================
    def _load_data(self,
                   hdf5_path: str,
                   max_jets: Optional[int],
                   event_mask: Optional[int],
                   valid_subset: bool,
                   sparse_format: bool):

        with H5File(hdf5_path, 'r') as file:
            # Take only the event with the specified number of top quarks if specified
            num_top_quarks = file['N_match_top_in_event'][:]
            if event_mask is None:
                tt_event_mask = np.ones_like(num_top_quarks, dtype=np.bool)
            else:
                tt_event_mask = (num_top_quarks == event_mask)

            self.num_top_quarks = num_top_quarks[tt_event_mask].astype(np.int)

            # Select the input features
            source_indices = ["jet_pt", "jet_mass", "jet_phi", "jet_eta", "jet_btag"]
            parton_indices = ["parton_pt", "parton_mass", "parton_phi", "parton_eta"]

            # Find the largest event in the dataset for padding
            if max_jets is None:
                max_jets = max(map(len, file["jet_mass"]))

            # Compute the shape parameters for the dataset
            self.num_samples = num_samples = tt_event_mask.sum()
            self.input_dim = len(source_indices)
            self.max_jets = max_jets

            # Create storage for data

            # Padding mask to indicate if a vector is in the event or a padded zero vector
            self.mask = torch.zeros((num_samples, max_jets), dtype=torch.bool)

            # Mask indicating if the vector is part of the partons
            self.subset_mask = torch.zeros((num_samples, max_jets), dtype=torch.bool)

            # The original jet vectors
            self.source = torch.zeros((num_samples, max_jets, self.input_dim), dtype=torch.float32)

            # The pre-selected parton vectors
            self.partons = torch.zeros((num_samples, 6, len(parton_indices)), dtype=torch.float32)

            # Output target storage
            targets = OrderedDict({
                "qq": torch.zeros((num_samples, max_jets, max_jets), dtype=torch.uint8),
                "b": torch.zeros((num_samples, max_jets, 1), dtype=torch.uint8),
                "qq_bar": torch.zeros((num_samples, max_jets, max_jets), dtype=torch.uint8),
                "b_bar": torch.zeros((num_samples, max_jets, 1), dtype=torch.uint8)
            })

            target_barcodes = {
                "qq": 20,
                "b": 17,
                "qq_bar": 40,
                "b_bar": 34
            }

            # Gather the HDF5 datasets for each index
            source_data = zip(*(file[index][tt_event_mask] for index in source_indices))
            parton_data = zip(*(file[index][:][tt_event_mask] for index in parton_indices))
            target_data = file["jet_barcode"][tt_event_mask]

            # Load in the data
            print(f"Reading data from hdf5 file: {hdf5_path}")
            print("This takes a long time, future runs will load from cache.")
            for i, (source, parton, barcodes) in tqdm(enumerate(zip(source_data, parton_data, target_data)), total=num_samples):
                source = torch.from_numpy(np.stack(source))
                parton = torch.from_numpy(np.stack(parton))
                num_jets = min(source.shape[1], max_jets)

                # Subset mask is where we have a parton-jet assignment (where not nan)
                subset_mask = ~np.isnan(barcodes)[:max_jets]
                self.subset_mask[i, :num_jets] = torch.from_numpy(subset_mask)

                # Remove the non-critical jets if enabled
                if valid_subset:
                    source = source[:, subset_mask]
                    barcodes = barcodes[subset_mask]
                    max_jets = min(np.sum(subset_mask), max_jets)

                # Load in the jet data
                self.source[i, :num_jets] = source.T[:num_jets]
                self.mask[i, :num_jets] = True
                self.partons[i] = parton.T

                # Load in the target data
                # We first load in the different particles into their own matrices
                for _, target, barcode in dict_zip(targets, target_barcodes):
                    idx = np.where(barcodes == barcode)[0]
                    try:
                        if barcode in {target_barcodes['qq'], target_barcodes['qq_bar']} and idx.shape[0] == 2:
                            a, b = idx
                            target[i, a, b] = 1
                            target[i, b, a] = 1
                        if barcode in {target_barcodes['b'], target_barcodes['b_bar']}:
                            target[i, idx[0]] = 1
                    except IndexError:
                        pass

            # TODO: This is current really inefficient because we construct the dense data first and then sparsify.

            # Combine the individual jet labels into the triplet targets
            self.targets = OrderedDict({
                't': targets['qq'].view(-1, max_jets, max_jets, 1) * targets['b'].view(-1, 1, 1, max_jets),
                't_bar': targets['qq_bar'].view(-1, max_jets, max_jets, 1) * targets['b_bar'].view(-1, 1, 1, max_jets)
            })

            # Convert to a sparse format for faster loading
            if sparse_format:
                self.targets = OrderedDict({
                    't': self.targets['t'].to_sparse(),
                    't_bar': self.targets['t_bar'].to_sparse(),
                })

    def _load_from_cache(self, cache_file):
        cache = torch.load(cache_file)
        self.source = cache["source"]
        self.partons = cache["partons"]
        self.targets = cache["targets"]
        self.mask = cache["mask"]
        self.subset_mask = cache["subset_mask"]
        self.num_top_quarks = cache["num_top_quarks"]

        self.num_samples, self.max_jets, self.input_dim = self.source.shape

    def _save_to_cache(self, cache_file):
        cache = {
            "source": self.source,
            "partons": self.partons,
            "targets": self.targets,
            "mask": self.mask,
            "subset_mask": self.subset_mask,
            "num_top_quarks": self.num_top_quarks
        }
        torch.save(cache, cache_file)

    def _limit_data_to_subset(self):
        self.max_jets = self.subset_mask.sum(1).max().item()
        self.source = self.source[:, :self.max_jets].contiguous()
        self.mask = self.mask[:, :self.max_jets].contiguous()
        for key, value in self.targets.items():
            self.targets[key] = value[:, :self.max_jets, :self.max_jets].contiguous()

    @staticmethod
    def contiguous_subset_sparse(tensor, subset_index):
        tensor = tensor.coalesce()

        shape = tensor.shape
        indices = tensor.indices().clone()
        values = tensor.values()

        max_index, min_index = subset_index.max(), subset_index.min()

        subset_index = (indices[0] <= max_index) & (indices[0] >= min_index)
        indices[0] -= min_index

        shape = (max_index - min_index + 1, *shape[1:])
        indices = indices.T[subset_index].T
        values = values[subset_index]

        return torch.sparse_coo_tensor(indices, values, size=shape)

    # =============================================================================================
    # Dataset Methods
    # =============================================================================================
    def normalize(self, mean=None, std=None):
        if mean is None:
            mean = self.source[self.mask].mean(0)
            std = self.source[self.mask].std(0)
            std[std < 1e-5] = 1

        self.mean = mean
        self.std = std

        return mean, std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = self.source[index].clone()
        mask = self.mask[index]

        if self.sparse_format:
            y = tuple(target[index].to_dense() for target in self.targets.values())
        else:
            y = tuple(target[index] for target in self.targets.values())

        # Normalize the input features in real-time
        if self.mean is not None:
            x[mask] -= self.mean
            x[mask] /= self.std

        return x, y, mask
