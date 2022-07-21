import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper
from cassle.losses.simclr import simclr_distill_loss_func
import torch.nn.functional as F


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

        def get_positive_logits(self, z1, z2):
            device = z1.device
            b = z1.size(0)
            z = torch.cat((z1, z2), dim=0)
            z = F.normalize(z, dim=-1)
            logits = torch.einsum("if, jf -> ij", z, z) / self.distill_temperature
            pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
            pos_mask[:, b:].fill_diagonal_(True)
            pos_mask[b:, :].fill_diagonal_(True)
            logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
            logits = torch.exp(logits)
            logits = logits / (logits * logit_mask).sum(1, keepdim=True)
            pos_logits = logits * pos_mask
            return pos_logits.sum(1)

        # def get_positive_logits(self, z1, z2):
        #     device = z1.device
        #     b = z1.size(0)
        #     z = torch.cat((z1, z2), dim=0)
        #     z = F.normalize(z, dim=-1)
        #     logits = torch.einsum("if, jf -> ij", z, z) / self.distill_temperature
        #     pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        #     pos_mask[:, b:].fill_diagonal_(True)
        #     pos_mask[b:, :].fill_diagonal_(True)
        #     logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
        #     logits = torch.exp(logits)
        #     logits = logits / (logits * logit_mask).sum(1, keepdim=True)
        #     pos_logits = logits * pos_mask
        #     return pos_logits

        def get_valid_mask(self, z1, z2, frozen_z1, frozen_z2):
            pos_logits_new = self.get_positive_logits(z1, z2)
            pos_logits_old = self.get_positive_logits(frozen_z1, frozen_z2)
            b = z1.size(0)
            mask1 = pos_logits_old[:b] > pos_logits_new[:b]
            mask2 = pos_logits_old[b:2 * b] > pos_logits_new[b:2 * b]
            valid_mask = mask1 | mask2
            return valid_mask

        # def get_valid_mask(self, z1, z2, frozen_z1, frozen_z2):
        #     pos_logits_new = self.get_positive_logits(z1, z2)
        #     pos_logits_old = self.get_positive_logits(frozen_z1, frozen_z2)
        #     valid_mask=pos_logits_old>=pos_logits_new
        #     return valid_mask

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out["z"]
            frozen_z1, frozen_z2 = out["frozen_z"]
            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)

            valid_mask=self.get_valid_mask(p1,p2,frozen_z1,frozen_z2)
            valid_mask=valid_mask.repeat(2)
            # p1=p1[valid_mask]
            # p2=p2[valid_mask]
            # frozen_z1=frozen_z1[valid_mask]
            # frozen_z2 = frozen_z2[valid_mask]
            self.log("valid_sample", sum(valid_mask), on_epoch=True, sync_dist=True)
            if sum(valid_mask) >0:
                distill_loss = (
                                       simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2, self.distill_temperature,valid_pos=valid_mask)
                                       + simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2, self.distill_temperature,valid_pos=valid_mask)
                               ) / 2
            else:
                distill_loss = 0.

            self.log("train_contrastive_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            return out["loss"] + self.distill_lamb * distill_loss

    return ContrastiveDistillWrapper
