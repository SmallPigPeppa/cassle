import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper
from cassle.losses.simclr import simclr_distill_loss_func


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
            self.att0_predictor = nn.Sequential(
                nn.Linear(output_dim, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, output_dim),
            )
            self.avgpool0 = nn.AdaptiveAvgPool2d((8, 8))
            self.att1_predictor = nn.Sequential(
                nn.Linear(output_dim, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, output_dim),
            )
            self.avgpool1 = nn.AdaptiveAvgPool2d((4, 4))
            self.att2_predictor = nn.Sequential(
                nn.Linear(output_dim, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, output_dim),
            )
            self.avgpool2 = nn.AdaptiveAvgPool2d((2, 2))

            self.att3_predictor = nn.Sequential(
                nn.Linear(output_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, output_dim),
            )
            self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
            self.att_predictor=[self.att0_predictor,self.att1_predictor,self.att2_predictor,self.att3_predictor]
            self.avgpool=[self.avgpool0,self.avgpool1,self.avgpool2,self.avgpool3]

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

        # def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        #     out = super().training_step(batch, batch_idx)
        #     z1, z2 = out["z"]
        #     frozen_z1, frozen_z2 = out["frozen_z"]
        #
        #     p1 = self.distill_predictor(z1)
        #     p2 = self.distill_predictor(z2)
        #
        #     distill_loss = (
        #         simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2, self.distill_temperature)
        #         + simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2, self.distill_temperature)
        #     ) / 2
        #
        #     self.log("train_contrastive_distill_loss", distill_loss, on_epoch=True, sync_dist=True)
        #
        #     return out["loss"] + self.distill_lamb * distill_loss

        # att0_1, att0_2 = attentions1[0], attentions2[0]
        # att1_1, att1_2 = attentions1[1], attentions2[1]
        # att2_1, att2_2 = attentions1[2], attentions2[2]
        # att3_1, att3_2 = attentions1[3], attentions2[3]
        # frozen_att0_1, frozen_att0_2 = frozen_attentions1[0], frozen_attentions2[0]
        # frozen_att1_1, frozen_att1_2 = frozen_attentions1[1], frozen_attentions2[1]
        # frozen_att2_1, frozen_att2_2 = frozen_attentions1[2], frozen_attentions2[2]
        # frozen_att3_1, frozen_att3_2 = frozen_attentions1[3], frozen_attentions2[3]
        # att0_1 = self.avgpool0(att0_1)
        # att0_1 = torch.flatten(att0_1, 1)
        # att0_2 = self.avgpool0(att0_2)
        # att0_2 = torch.flatten(att0_2, 1)
        #
        # att0_1 = self.att0_predictor(att0_1)
        # att0_2 = self.att0_predictor(att0_2)
        # att1_1 = self.att1_predictor(att1_1)
        # att1_2 = self.att1_predictor(att1_2)
        # att2_1 = self.att0_predictor(att2_1)
        # att2_2 = self.att0_predictor(att2_2)
        # att3_1 = self.att0_predictor(att3_1)
        # att3_2 = self.att0_predictor(att3_2)
        #
        # att0_loss = (
        #                     simclr_distill_loss_func(att0_1, att0_2, frozen_att0_1, frozen_att0_2,
        #                                              self.distill_temperature)
        #                     + simclr_distill_loss_func(frozen_att0_1, frozen_att0_2, att0_1, att0_2,
        #                                                self.distill_temperature)
        #             ) / 2
        # att1_loss = (
        #                     simclr_distill_loss_func(att1_1, att1_2, frozen_att1_1, frozen_att1_2,
        #                                              self.distill_temperature)
        #                     + simclr_distill_loss_func(frozen_att1_1, frozen_att1_2, att1_1, att1_2,
        #                                                self.distill_temperature)
        #             ) / 2
        # att2_loss = (
        #                     simclr_distill_loss_func(att2_1, att2_2, frozen_att2_1, frozen_att2_2,
        #                                              self.distill_temperature)
        #                     + simclr_distill_loss_func(frozen_att2_1, frozen_att2_2, att2_1, att2_2,
        #                                                self.distill_temperature)
        #             ) / 2
        # att3_loss = (
        #                     simclr_distill_loss_func(att3_1, att3_2, frozen_att3_1, frozen_att3_2,
        #                                              self.distill_temperature)
        #                     + simclr_distill_loss_func(frozen_att3_1, frozen_att3_2, att3_1, att3_2,
        #                                                self.distill_temperature)
        #             ) / 2
        #
        # self.log("att0_loss", att0_loss, on_epoch=True, sync_dist=True)
        # self.log("att1_loss", att1_loss, on_epoch=True, sync_dist=True)
        # self.log("att2_loss", att2_loss, on_epoch=True, sync_dist=True)
        # self.log("att3_loss", att3_loss, on_epoch=True, sync_dist=True)

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out["z"]
            frozen_z1, frozen_z2 = out["frozen_z"]

            z1 = self.distill_predictor(z1)
            z2 = self.distill_predictor(z2)

            distill_loss = (
                                   simclr_distill_loss_func(z1, z2, frozen_z1, frozen_z2, self.distill_temperature)
                                   + simclr_distill_loss_func(frozen_z1, frozen_z2, z1, z2, self.distill_temperature)
                           ) / 2

            self.log("train_contrastive_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            attentions1, attentions2 = out["attentions"]
            frozen_attentions1, frozen_attentions2 = out["frozen_attentions"]
            att_loss=[]
            for i,(att1, att2,frozen_att1,frozen_att2) in enumerate(zip(attentions1, attentions2,frozen_attentions1, frozen_attentions2)):
                att1=self.avgpool[i](att1)
                att2 = self.avgpool[i](att2)
                att1=torch.flatten(att1,1)
                att2 = torch.flatten(att2, 1)
                frozen_att1=self.avgpool[i](frozen_att1)
                frozen_att2 = self.avgpool[i](frozen_att2)
                frozen_att1=torch.flatten(frozen_att1,1)
                frozen_att2 = torch.flatten(frozen_att2, 1)

                att1=self.att_predictor[i](att1)
                att2 = self.att_predictor[i](att2)

                att_loss.append((
                                   simclr_distill_loss_func(att1, att2, frozen_att1, frozen_att2, self.distill_temperature)
                                   + simclr_distill_loss_func(frozen_att1, frozen_att2, att1, att2, self.distill_temperature)
                           ) / 2)
                self.log(f"att{i}_loss", att_loss[i], on_epoch=True, sync_dist=True)
            return out["loss"] + self.distill_lamb * (distill_loss+att_loss[0]+att_loss[1]+att_loss[2]+att_loss[3])

    return ContrastiveDistillWrapper
