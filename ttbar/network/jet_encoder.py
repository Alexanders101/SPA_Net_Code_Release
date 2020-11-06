from typing import Tuple

import torch
from torch import nn, Tensor

from ttbar.network.network_utils import create_linear_stack
from ttbar.options import Options


class JetEncoder(nn.Module):
    def __init__(self, options: Options, input_dim: int, transformer_options: Tuple[int, int, int, float, str]):
        super(JetEncoder, self).__init__()

        self.options = options

        self.embedding = self.create_embedding_layers(input_dim)

        self.encoder_layer = nn.TransformerEncoderLayer
        self.encoder = nn.TransformerEncoder(self.encoder_layer(*transformer_options), options.num_encoder_layers)

    def create_embedding_layers(self, input_dim):
        current_embedding_dim = self.options.initial_embedding_dim
        embedding_layers = create_linear_stack(input_dim, current_embedding_dim, self.options)

        for i in range(self.options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= self.options.hidden_dim:
                break

            embedding_layers.extend(create_linear_stack(current_embedding_dim, next_embedding_dim, self.options))
            current_embedding_dim = next_embedding_dim

        embedding_layers.extend(create_linear_stack(current_embedding_dim, self.options.hidden_dim, self.options))

        return nn.Sequential(*embedding_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, max_jets, input_dim = x.shape

        # Setup variants of mask
        padding_mask = ~mask

        sequence_mask = mask.view(batch_size, max_jets, 1).transpose(0, 1).contiguous()
        if not self.options.mask_sequence_vectors:
            sequence_mask = torch.ones_like(sequence_mask)

        # Perform embedding on all of the vectors uniformly
        hidden = self.embedding(x.view(-1, input_dim))
        hidden = hidden.view(batch_size, max_jets, self.options.hidden_dim)

        # Reshape vector to have time axis first for transformer input
        hidden = hidden.transpose(0, 1) * sequence_mask

        # Primary central transformer
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask) * sequence_mask

        return hidden, padding_mask, sequence_mask