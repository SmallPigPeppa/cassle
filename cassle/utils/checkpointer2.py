import json
import os
import random
import string
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional,Union
import torch
from copy import deepcopy
from pytorch_lightning.utilities.types import _METRIC
from pytorch_lightning.utilities.warnings import WarningCache
warning_cache = WarningCache()
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


def random_string(letter_count=4, digit_count=4):
    tmp_random = random.Random(time.time())
    rand_str = "".join((tmp_random.choice(string.ascii_lowercase) for x in range(letter_count)))
    rand_str += "".join((tmp_random.choice(string.digits) for x in range(digit_count)))
    rand_str = list(rand_str)
    tmp_random.shuffle(rand_str)
    return "".join(rand_str)


class Checkpointer(Callback):
    def __init__(
        self,
        args: Namespace,
        logdir: Union[str, Path] = Path("trained_models"),
        frequency: int = 1,
        keep_previous_checkpoints: bool = False,
    ):
        """Custom checkpointer callback that stores checkpoints in an easier to access way.

        Args:
            args (Namespace): namespace object containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            frequency (int, optional): number of epochs between each checkpoint. Defaults to 1.
            keep_previous_checkpoints (bool, optional): whether to keep previous checkpoints or not.
                Defaults to False.
        """

        super().__init__()

        assert "task" not in args.name

        self.args = args
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.keep_previous_checkpoints = keep_previous_checkpoints
        self.monitor='val_acc1'
        self.best_monitor=0.

    @staticmethod
    def add_checkpointer_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("checkpointer")
        parser.add_argument("--checkpoint_dir", default=Path("trained_models"), type=Path)
        parser.add_argument("--checkpoint_frequency", default=1, type=int)
        return parent_parser

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            if os.path.exists(self.logdir):
                existing_versions = set(os.listdir(self.logdir))
            else:
                existing_versions = set()
            version = "offline-" + random_string()
            while version in existing_versions:
                version = "offline-" + random_string()
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = self.logdir / version
            self.ckpt_placeholder = f"{self.args.name}" + "-task{}-ep={}" + f"-{version}.ckpt"
        self.best_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def save_args(self, trainer: pl.Trainer):
        """Stores arguments into a json file.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero:
            args = vars(self.args)
            self.json_path = self.path / "args.json"
            json.dump(args, open(self.json_path, "w"), default=lambda o: "<not serializable>")

    def save(self, trainer: pl.Trainer):
        """Saves current checkpoint.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero and not trainer.sanity_checking:
            epoch = trainer.current_epoch  # type: ignore
            task_idx = getattr(self.args, "task_idx", "_all")
            ckpt = self.path / self.ckpt_placeholder.format(task_idx, epoch)
            trainer.save_checkpoint(ckpt)

            # if self.last_ckpt and self.last_ckpt != ckpt and not self.keep_previous_checkpoints:
            #     if os.path.exists(self.last_ckpt):
            #         os.remove(self.last_ckpt)

            if self.best_ckpt and self.best_ckpt != ckpt and not self.keep_previous_checkpoints:
                if os.path.exists(self.best_ckpt):
                    os.remove(self.best_ckpt)

            # with open(self.logdir / "last_checkpoint.txt", "w") as f:
            #     f.write(str(ckpt) + "\n" + str(self.json_path))

            with open(self.logdir / "last_checkpoint.txt", "w") as f:
                f.write(str(ckpt) + "\n" + str(self.json_path))

            # self.last_ckpt = ckpt
            self.best_ckpt = ckpt

    def on_train_start(self, trainer: pl.Trainer, _):
        """Executes initial setup and saves arguments.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """Tries to save current checkpoint at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        # epoch = trainer.current_epoch  # type: ignore
        # if epoch % self.frequency == 0:
        #     self.save(trainer)
        monitor_candidates = self._monitor_candidates(trainer)
        current_monitor = monitor_candidates.get(self.monitor)
        print('#################################current_monitor#################################')
        print(current_monitor)
        print('###################################################################################')
        if current_monitor >= self.best_monitor:
            self.best_monitor=current_monitor
            self.save(trainer)


    def _monitor_candidates(self, trainer: "pl.Trainer") -> Dict[str, _METRIC]:
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = (
            epoch.int() if isinstance(epoch, torch.Tensor) else torch.tensor(trainer.current_epoch)
        )
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, torch.Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates

