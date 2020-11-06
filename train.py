from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl

from ttbar.network import QuarkTripletNetwork
from ttbar.options import Options


def main(input_file: str, checkpoint: Optional[str], epochs: Optional[int], fp16: bool):
    hparams = Options(input_file)

    print("Options")
    print("=" * 60)
    for key, val in hparams.__dict__.items():
        print(f"{key:20}: {val}")

    model = QuarkTripletNetwork(hparams)
    distributed_backend = 'ddp' if hparams.num_gpu > 1 else None
    trainer = pl.Trainer(max_epochs=hparams.epochs if epochs is None else epochs,
                         resume_from_checkpoint=checkpoint,
                         distributed_backend=distributed_backend,
                         gpus=hparams.num_gpu,
                         precision=16 if fp16 else 32)

    print(f"Training Version {trainer.logger.version}")
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-i", "--input_file", type=str, default="./data/event_records_training.h5",
                        help="Input file containing training data.")

    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load from")

    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Override the number of epochs to train for.")

    parser.add_argument("-fp16", action="store_true",
                        help="Use AMP for training.")

    main(**parser.parse_args().__dict__)