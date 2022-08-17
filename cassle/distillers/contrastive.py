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

        # def pl_loss(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        #     z = self.frozen_projector(feats)
        #     z_centers = self.frozen_projector(self.feats_centers)
        #     z = z.reshape(-1, 1, 256)
        #     labels = labels - (45 + self.current_task_idx * 5)
        #     z_centers = z_centers.reshape(1, -1, 256)
        #     cosine_distance = self.cosine(z, z_centers.detach())
        #     inclass_distance = torch.index_select(cosine_distance, dim=1, index=labels)
        #     inclass_distance = torch.diagonal(inclass_distance)
        #     pl_loss = torch.mean(inclass_distance)
        #     return pl_loss

        def pl_loss(self, z_centers: torch.Tensor, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            z = z.reshape(-1, 1, 256)
            labels = labels - (45 + self.current_task_idx * 5)
            z_centers = z_centers.reshape(1, -1, 256)
            cosine_distance = self.cosine(z, z_centers.detach())
            inclass_distance = torch.index_select(cosine_distance, dim=1, index=labels)
            inclass_distance = torch.diagonal(inclass_distance)
            pl_loss = torch.mean(inclass_distance)
            return pl_loss

        def groupby_mean(self, z, labels):
            """Group-wise average for (sparse) grouped tensors

            Args:
                z (torch.Tensor): values to average (# samples, latent dimension)
                labels (torch.LongTensor): labels for embedding parameters (# samples,)

            Returns:
                result (torch.Tensor): (# unique labels, latent dimension)
                new_labels (torch.LongTensor): (# unique labels,)

            Examples:
                >>> samples = torch.Tensor([
                                     [0.15, 0.15, 0.15],    #-> group / class 1
                                     [0.2, 0.2, 0.2],    #-> group / class 3
                                     [0.4, 0.4, 0.4],    #-> group / class 3
                                     [0.0, 0.0, 0.0]     #-> group / class 0
                              ])
                >>> labels = torch.LongTensor([1, 5, 5, 0])
                >>> result, new_labels = groupby_mean(samples, labels)

                >>> result
                tensor([[0.0000, 0.0000, 0.0000],
                    [0.1500, 0.1500, 0.1500],
                    [0.3000, 0.3000, 0.3000]])

                >>> new_labels
                tensor([0, 1, 5])
            """
            uniques = labels.unique().tolist()
            labels = labels.tolist()

            key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
            val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

            labels = torch.LongTensor(list(map(key_val.get, labels)))

            labels = labels.view(labels.size(0), 1).expand(-1, z.size(1))

            unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
            result = torch.zeros_like(unique_labels, dtype=torch.float, device=self.device).scatter_add_(0, labels, z)
            result = result / labels_count.float().unsqueeze(1)
            new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))

            _, order_index = new_labels.sort()
            return result[order_index]

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            # z1, z2 = out["z"]
            frozen_z1, frozen_z2 = out["frozen_z"]
            # frozen_feats1,frozen_feats2=out["frozen_feats"]
            # frozen_z1=self.projector(frozen_feats1)
            # frozen_z2=self.projector(frozen_feats2)
            feats1, feats2 = out["feats"]
            p1 = self.frozen_projector(feats1)
            p2 = self.frozen_projector(feats2)
            _, *_, target = batch[f"task{self.current_task_idx}"]
            z_centers = self.groupby_mean(z=torch.vstack((frozen_z1, frozen_z2)), labels=target.repeat(2))
            pl_loss = (self.pl_loss(z_centers=z_centers, z=p1, labels=target) + self.pl_loss(z_centers=z_centers, z=p2,
                                                                                             labels=target)) / 2

            self.log("pl_loss", pl_loss, on_epoch=True, sync_dist=True)
            return out["loss"] - 0.0 * pl_loss

        # def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        #     out = super().training_step(batch, batch_idx)
        #     # z1, z2 = out["z"]
        #     frozen_z1, frozen_z2 = out["frozen_z"]
        #     # frozen_feats1,frozen_feats2=out["frozen_feats"]
        #     # frozen_z1=self.projector(frozen_feats1)
        #     # frozen_z2=self.projector(frozen_feats2)
        #     feats1, feats2 = out["feats"]
        #     _, *_, target = batch[f"task{self.current_task_idx}"]
        #     pl_loss = (self.pl_loss(feats=feats1, labels=target) + self.pl_loss(feats=feats2, labels=target)) / 2
        #     # p1 = self.frozen_projector(feats1)
        #     # p2 = self.frozen_projector(feats2)
        #     # # p1 = self.distill_predictor(z1)
        #     # # p2 = self.distill_predictor(z2)
        #     # # p1 = z1
        #     # # p2 = z2
        #     #
        #     # distill_loss = (
        #     #                        simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2, self.distill_temperature)
        #     #                        + simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2, self.distill_temperature)
        #     #                ) / 2
        #     #
        #     # self.log("train_contrastive_distill_loss", distill_loss, on_epoch=True, sync_dist=True)
        #     self.log("pl_loss", pl_loss, on_epoch=True, sync_dist=True)
        #     return out["loss"] - 0.1 * pl_loss
        #
        #     # return out["loss"] + self.distill_lamb * distill_loss

    return ContrastiveDistillWrapper
