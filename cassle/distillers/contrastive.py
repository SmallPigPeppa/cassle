import argparse
from typing import Any, Callable, Dict, List, Sequence, Tuple
import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper
from cassle.losses.simclr import simclr_distill_loss_func
from cassle.utils.momentum import MomentumUpdater, initialize_momentum_params

def contrastive_distill_wrapper(Method=object):
    class ContrastiveDistillWrapper(base_distill_wrapper(Method)):
        def __init__(
            self,
            distill_lamb: float,
            distill_proj_hidden_dim: int,
            distill_temperature: float,
            **kwargs
        ):
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            self.distill_temperature = distill_temperature
            output_dim = kwargs["output_dim"]
            self.distill_predictor = nn.Sequential(
                nn.Linear(output_dim, distill_proj_hidden_dim),
                nn.BatchNorm1d(distill_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(distill_proj_hidden_dim, output_dim),
            )
            # self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)
            self.momentum_updater = MomentumUpdater(0.99, 1.0)
        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--distill_lamb", type=float, default=1)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)
            parser.add_argument("--distill_temperature", type=float, default=0.2)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": self.distill_predictor.parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out["z"]
            frozen_z1, frozen_z2 = out["frozen_z"]

            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)

            distill_loss = (
                simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2, self.distill_temperature)
                + simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2, self.distill_temperature)
            ) / 2

            self.log("train_contrastive_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            return out["loss"] + self.distill_lamb * distill_loss

        def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
            """Performs the momentum update of momentum pairs using exponential moving average at the
            end of the current training step if an optimizer step was performed.

            Args:
                outputs (Dict[str, Any]): the outputs of the training step.
                batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                    [X] is a list of size self.num_crops containing batches of images.
                batch_idx (int): index of the batch.
            """

            if self.trainer.global_step > self.last_step:
                # update momentum encoder and projector
                # self.frozen_encoder = deepcopy(self.encoder)
                # self.frozen_projector = deepcopy(self.projector)
                # [(self.encoder, self.momentum_encoder)]
                momentum_pairs = [(self.encoder, self.frozen_encoder),(self.projector,self.frozen_projector)]
                for mp in momentum_pairs:
                    self.momentum_updater.update(*mp)
                # log tau momentum
                self.log("tau", self.momentum_updater.cur_tau)
                # update tau
                self.momentum_updater.update_tau(
                    cur_step=self.trainer.global_step * self.trainer.accumulate_grad_batches,
                    max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
                )
            self.last_step = self.trainer.global_step

        def on_train_start(self):
            """Resets the step counter at the beginning of training."""
            super().on_train_start()
            self.last_step = 0

    return ContrastiveDistillWrapper