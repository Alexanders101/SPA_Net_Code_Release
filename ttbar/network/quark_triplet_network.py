from typing import Tuple

import torch
from torch import Tensor

from ttbar.network.jet_encoder import JetEncoder
from ttbar.network.triplet_decoder import TripletDecoder
from ttbar.options import Options
from ttbar.network.quark_base_network import QuarkBaseNetwork

PredictType = Tuple[Tensor, Tensor]
TargetType = Tuple[Tensor, Tensor]


class QuarkTripletNetwork(QuarkBaseNetwork):
    def __init__(self, options: Options, client=None, trial=None):
        super().__init__(options, client, trial)

        self.hidden_dim = options.hidden_dim
        self.num_jets = self.training_dataset.max_jets

        transformer_options = (options.hidden_dim,
                               options.num_attention_heads,
                               options.hidden_dim,
                               options.dropout,
                               options.transformer_activation)

        self.encoder = JetEncoder(options, self.training_dataset.input_dim, transformer_options)

        self.left_decoder = TripletDecoder(options, transformer_options)
        self.right_decoder = TripletDecoder(options, transformer_options)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> PredictType:
        # Extract features from data using transformer
        hidden, padding_mask, sequence_mask = self.encoder(x, mask)

        left_output = self.left_decoder(hidden, padding_mask, sequence_mask)
        right_output = self.right_decoder(hidden, padding_mask, sequence_mask)

        return left_output, right_output

    @staticmethod
    def negative_log_likelihood(prediction, target, eps=1e-6):
        """ Loss term for triplet prediction. """
        return -torch.sum(torch.log(prediction + eps) * target, dim=(1, 2, 3))

    @staticmethod
    def symmetric_cross_entropy(x, y, eps=1e-6):
        """ Loss term for seperating the two output heads. """
        H_xy = x * torch.log(y + eps)
        H_yx = y * torch.log(x + eps)

        return -torch.sum(H_xy + H_yx, dim=(1, 2, 3))

    def loss(self, predictions: PredictType, targets: TargetType) -> Tensor:
        left_prediction, right_prediction = predictions
        left_target, right_target = targets

        batch_size, max_jets, _, _ = left_prediction.shape

        total_loss = self.negative_log_likelihood(left_prediction, left_target)
        total_loss = total_loss + self.negative_log_likelihood(right_prediction, right_target)

        if self.options.triplet_difference_loss > 0:
            difference_loss = self.symmetric_cross_entropy(left_prediction, right_prediction)
            total_loss = total_loss - difference_loss * self.options.triplet_difference_loss

        return total_loss

    @staticmethod
    def swap_antiparticle_targets(targets: TargetType) -> TargetType:
        """ Charge symmetry for the targets. """
        return targets[1], targets[0]

    def training_step(self, batch, batch_nb):
        x, targets, mask = batch

        predictions = self.forward(x, mask)
        swapped_targets = self.swap_antiparticle_targets(targets)

        regular_loss = self.loss(predictions, targets)
        swapped_loss = self.loss(predictions, swapped_targets)

        combined_loss = regular_loss
        if self.options.combine_pair_loss == "min":
            combined_loss = torch.min(regular_loss, swapped_loss)
        elif self.options.combine_pair_loss == "sum":
            combined_loss = (regular_loss + swapped_loss) / 2.0

        loss = torch.mean(combined_loss)

        if torch.isnan(loss):
            raise ValueError("Training loss has diverged.")

        self.log("train_loss", loss)

        return loss

    @staticmethod
    def accuracy(predictions: PredictType, targets: TargetType) -> Tuple[Tensor, Tensor]:
        """ Compute single top and eventy accuracy for a batch. """
        left_predictions = predictions[0].clone()
        right_predictions = predictions[1].clone()

        left_targets = targets[0].clone()
        right_targets = targets[1].clone()

        batch_size, max_jets, _, _ = left_predictions.shape

        # Zero out the lower triangle to make accuracy calculation easier
        # Both targets and predictions should be symmetric anyway
        for i in range(max_jets):
            for j in range(i):
                left_predictions[:, i, j, :] = 0
                right_predictions[:, i, j, :] = 0
                left_targets[:, i, j, :] = 0
                right_targets[:, i, j, :] = 0

        left_targets = left_targets.view(batch_size, -1).argmax(1)
        right_targets = right_targets.view(batch_size, -1).argmax(1)

        left_predictions = left_predictions.view(batch_size, -1).argmax(1)
        right_predictions = right_predictions.view(batch_size, -1).argmax(1)

        left_accuracy = left_targets == left_predictions
        right_accuracy = right_targets == right_predictions

        either_accuracy = left_accuracy | right_accuracy
        both_accuracy = left_accuracy & right_accuracy

        return either_accuracy, both_accuracy

    def validation_step(self, batch, batch_idx):
        x, targets, mask = batch
        predictions = self.forward(x, mask)

        swapped_targets = self.swap_antiparticle_targets(targets)

        either_accuracy, both_accuracy = self.accuracy(predictions, targets)
        swapped_either_accuracy, swapped_both_accuracy = self.accuracy(predictions, swapped_targets)

        either_accuracy = either_accuracy | swapped_either_accuracy
        both_accuracy = both_accuracy | swapped_both_accuracy

        both_accuracy = both_accuracy.float().mean()
        either_accuracy = either_accuracy.float().mean()

        self.log("triplet_both_accuracy", both_accuracy)
        self.log("triplet_either_accuracy", either_accuracy)

        return {"triplet_both_accuracy": both_accuracy, "triplet_either_accuracy": either_accuracy}

    def validation_epoch_end(self, outputs):
        average_triplet_both_accuracy = torch.mean(torch.stack([x['triplet_both_accuracy'] for x in outputs]))
        average_triplet_either_accuracy = torch.mean(torch.stack([x['triplet_either_accuracy'] for x in outputs]))

        self.commit_sherpa(average_triplet_both_accuracy)

        return {"triplet_both_accuracy": average_triplet_both_accuracy,
                "triplet_either_accuracy": average_triplet_either_accuracy}


