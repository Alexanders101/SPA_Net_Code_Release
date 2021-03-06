from argparse import Namespace


class Options(Namespace):
    def __init__(self, training_file: str, testing_file: str = ""):
        super(Options, self).__init__()

        # =========================================================================================
        # Network Architecture
        # =========================================================================================

        # Dimensions used internally by all hidden layers / transformers
        self.hidden_dim = 128

        # Dimensions of the first embedding layer
        self.initial_embedding_dim = 8

        # Maximum Number of increasingly sized embedding layers to add between the features and the encoder
        # Set to a large value to always fill the powers of 2
        self.num_embedding_layers = 10

        # Number of encoder layers for the main transformer
        self.num_encoder_layers = 6

        # Number of feed forward layers to add to branch heads
        # Set to 0 to disable branch embedding layers
        self.num_branch_embedding_layers = 4

        # Number of encoder layers for each of the quark branch transformers
        # Set to 0 to disable branch encoder layers
        self.num_branch_encoder_layers = 4

        # Number of heads for multi-head attention, used in encoder layers
        self.num_attention_heads = 4

        # Activation function for transformer, 'relu' or 'gelu'
        self.transformer_activation = 'gelu'

        # Whether or not to use PreLU activation on linear (embedding) layers
        # Otherwise a regular relu will be used
        self.linear_prelu_activation = True

        # Whether or not to apply batch norm on linear (embedding) layers
        self.linear_batch_norm = True

        # The beta term for the symmetric cross-entropy loss
        self.triplet_difference_loss = 0.2

        # =========================================================================================
        # Dataset Options
        # =========================================================================================

        # LHE files containing the collisions
        self.event_file = training_file
        self.testing_file = testing_file

        # Maximum number of jets in the data
        # Set to None to automatically determine from data
        self.num_jets = 20

        # Whether or not to take the 6 valid jets and ignore the garbage jets
        self.valid_subset = False

        # Percent of data to use for training vs validation
        self.train_test_split = 0.8

        # Training batch size
        self.batch_size = 1024

        # =========================================================================================
        # Training Options
        # =========================================================================================

        # Whether or not to mask vectors not in the events during the forward pass
        # Should most-definitely be True, but this is here for testing purposes
        self.mask_sequence_vectors = True

        # Whether we should combine the two possible targets: swapped and not-swapped
        # Current options are None, 'min', and 'sum
        # If None, then we will only use the exact target ordering

        # self.combine_pair_loss = None
        self.combine_pair_loss = 'min'
        # self.combine_pair_loss = 'sum'

        # The optimizer to use for training the network
        # This must be a valid class in torch.optim or nvidia apex with 'apex' prefix
        # See `ttbar.network.quark_base_network.py` -> `configure_optimizers()` for more information.
        self.optimizer = "apex_lamb"

        # Optimizer learning rate
        self.learning_rate = 0.001

        # Optimizer l2 penalty scale
        self.l2_penalty = 9.371e-05

        # Optional dropout added to all transformer layers
        self.dropout = 0.0

        # Total number of epochs to train for
        self.epochs = 500

        # Total number of GPUs to use for training
        self.num_gpu = 4

        # Number of processes to spawn for data collection
        # This is per-gpu! So if we use 4 gpus, then 64 workers will be spawned.
        self.num_dataloader_workers = 16

        # Extra parameters that are used by sherpa for managing distribution hyperparameter optimization.
        self.usable_gpus = ''

        self.trial_time = ''

        self.trial_output_dir = './test_output'

    def update_hparams(self, new_params):
        integer_options = {key for key, val in self.__dict__.items() if isinstance(val, int)}
        boolean_options = {key for key, val in self.__dict__.items() if isinstance(val, bool)}
        for key, value in new_params.items():
            if key in integer_options:
                setattr(self, key, int(value))
            elif key in boolean_options:
                setattr(self, key, bool(value))
            else:
                setattr(self, key, value)
