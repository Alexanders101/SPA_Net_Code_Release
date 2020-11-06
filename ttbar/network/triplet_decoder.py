from typing import Tuple

import math
import torch
from torch import nn, Tensor
from opt_einsum import contract as einsum

from ttbar.options import Options
from ttbar.network.network_utils import masked_softmax, create_linear_layers


class TripletSelfAttention(torch.jit.ScriptModule):
    def __init__(self, features: int, bias: bool = False) -> None:
        super(TripletSelfAttention, self).__init__()

        self.features = features
        self.weight = nn.Parameter(torch.randn(features, features, features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce that the weight must be symmetric along n,m and independent along l
        symmetric_weight = einsum('nij,mij,lkk->nml', self.weight, self.weight, self.weight, backend='torch')
        symmetric_weight = symmetric_weight / (self.features * self.features)

        # Perform the generalized matrix multiplication operation.
        output = einsum("bni,bmj,blk,ijk->bnml", x, x, x, symmetric_weight, backend='torch')
        output = output / math.sqrt(self.features)

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return 'features={}, bias={}'.format(self.features, self.bias is not None)


class BranchEncoder(nn.Module):
    def __init__(self, options: Options, transformer_options: Tuple[int, int, int, float, str]):
        super(BranchEncoder, self).__init__()
        self.options = options

        self.embedding = None
        if options.num_branch_embedding_layers > 0:
            self.embedding = create_linear_layers(options.num_branch_embedding_layers, options.hidden_dim, options)

        self.encoder = None
        if options.num_branch_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer
            self.encoder = nn.TransformerEncoder(encoder_layer(*transformer_options), options.num_branch_encoder_layers)

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        max_jets, batch_size, input_dim = x.shape

        # =========================================================================================
        # Flat Pre-embedding
        # =========================================================================================
        if self.embedding is not None:
            x = x.reshape(-1, input_dim)
            x = self.embedding(x)
            x = x.reshape(max_jets, batch_size, input_dim)
            x = x * sequence_mask

        # =========================================================================================
        # Transformer Encoder
        # =========================================================================================
        if self.encoder is not None:
            x = self.encoder(x, src_key_padding_mask=padding_mask)
            x = x * sequence_mask

        return x


class TripletDecoder(nn.Module):
    def __init__(self, options: Options, transformer_options: Tuple[int, int, int, float, str]):
        super(TripletDecoder, self).__init__()

        self.encoder = BranchEncoder(options, transformer_options)
        self.decoder = TripletSelfAttention(options.hidden_dim)

    @staticmethod
    def create_weights_mask(weights: Tensor, sequence_mask: Tensor) -> Tensor:
        batch_size, max_jets, max_jets, max_jets = weights.shape
        mask = torch.ones_like(weights)
        
        # Padding mask
        sequence_mask = sequence_mask.transpose(0, 1)
        mask = mask * sequence_mask.view(batch_size, 1, 1, max_jets)
        mask = mask * sequence_mask.view(batch_size, 1, max_jets, 1)
        mask = mask * sequence_mask.view(batch_size, max_jets, 1, 1)
        
        # Diagonal mask
        for i in range(max_jets):
            mask[:, i, i, :] = 0
            mask[:, i, :, i] = 0
            mask[:, :, i, i] = 0

        return mask

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        q = self.encoder(x, padding_mask, sequence_mask)

        weights = self.decoder(q.transpose(0, 1))
        mask = self.create_weights_mask(weights, sequence_mask)

        # Perform softmax over the entire upper triangle of the weights matrix
        batch_size, max_jets, max_jets, max_jets = weights.shape
        weights = weights.view(batch_size, -1)
        mask = mask.view(batch_size, -1)

        weights = masked_softmax(weights, mask)
        weights = weights.view(batch_size, max_jets, max_jets, max_jets)

        return weights
