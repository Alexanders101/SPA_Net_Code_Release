import torch
from torch import nn

from ttbar.options import Options


@torch.jit.script
def masked_softmax(x: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = 1,
                   eps: torch.Tensor = torch.tensor(1e-6, dtype=torch.float)):
    offset = x.max(dim, keepdim=True).values
    output = torch.exp(x - offset) * mask

    normalizing_sum = output.sum(dim, keepdim=True) + eps
    return output / normalizing_sum


def create_linear_stack(input_dim, output_dim, options: Options):
    layers = [nn.Linear(input_dim, output_dim)]

    if options.linear_prelu_activation:
        layers.append(nn.PReLU(output_dim))
    else:
        layers.append(nn.ReLU())

    if options.linear_batch_norm:
        layers.append(nn.BatchNorm1d(output_dim))

    if options.dropout > 0.0:
        layers.append(nn.Dropout(options.dropout))

    return layers


def create_linear_layers(num_layers: int, hidden_dim: int, options: Options):
    layers = []

    for _ in range(num_layers):
        layers.extend(create_linear_stack(hidden_dim, hidden_dim, options))

    return nn.Sequential(*layers)