# SPA-Net

*Permutationless Many-Jet Event Reconstruction with Symmetry Preserving Attention Networks*. \
https://arxiv.org/abs/2010.09206

# Requirements
- python >= 3.7
- h5py
- [pytorch](https://pytorch.org/) >= 1.6
- [pytorch-lightning](https://www.pytorchlightning.ai/) >= 1.0
- [parameter-sherpa](https://parameter-sherpa.readthedocs.io/en/latest/)
- [nvidia-apex for optimizers](https://nvidia.github.io/apex/optimizers.html)

# Training Instructions
1. Get the data release from the following link: **Data and pretrained model will be released soon.**
2. Modify `tbar/options.py` to match your system and data location.
3. Run `python train.py`
4. During trianing, metrics will be published into lightning_logs.
5. After training, weights will be available in lightning_logs.

# Code Structure
- `ttbar/options.py` Contains all of the hyperparameters and options used during training.
- `ttbar/dataset.py` Is reponsible for loading the HDF5 files that we extracted from madgraph root files.
- `ttbar/network/quark_triplet_network.py` Describes the main network architecture and training procedure.

# Citation
The current citation is to the arXiv preprint. 
This may be updated in the future.

```
@misc{fenton_2020_spatter,
      title={Permutationless Many-Jet Event Reconstruction with Symmetry Preserving Attention Networks}, 
      author={Michael James Fenton and Alexander Shmakov and Ta-Wei Ho and Shih-Chieh Hsu and Daniel Whiteson and Pierre Baldi},
      year={2020},
      eprint={2010.09206},
      archivePrefix={arXiv},
      primaryClass={hep-ex}
}
```