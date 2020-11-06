from abc import ABC
from typing import Optional

import pytorch_lightning as pl
import sherpa


class SherpaModule(pl.LightningModule, ABC):
    def __init__(self, hparams, client: Optional[sherpa.Client] = None, trial: Optional[sherpa.Trial] = None):
        super(SherpaModule, self).__init__()

        self.hparams = hparams

        # Sherpa Objects
        self.client = client
        self.trial = trial
        self.sherpa_iteration = 0

        # study => trial
        assert (not self.client) or self.trial

    def commit_sherpa(self, objective, context: Optional[dict] = None):
        if context is None:
            context = {}

        if self.client:
            self.sherpa_iteration += 1
            self.client.send_metrics(trial=self.trial,
                                     iteration=self.sherpa_iteration,
                                     objective=objective.item(),
                                     context={key: val.item() for key, val in context.items()})
