import math
from typing import Callable, Iterator

import lightning as L
from torch import tensor
import copy

from experiments.nets.baseunet import UNet
from experiments.utils import assert_eq
from itertools import repeat

Temperature = float | Iterator[float]


def default_teacher_temperature() -> Iterator[float]:
    i = 0
    while True:
        yield max(0.04 + 0.001 * i, 0.07)


def default_lambda_scheduler(total_step: int) -> Iterator[float]:
    lstart = 0.996
    lend = 1.0
    step = 0
    while True:
        yield lend + (lstart - lend) * (1 + math.cos(math.pi * total_step / step)) / 2

        step += 1


class DinoModule(L.LightningModule):
    def __init__(
        self,
        unet: UNet,
        temperature: tuple[Temperature, Temperature] = (
            0.06,
            default_teacher_temperature(),
        ),
        lambda_scheduler_fn: Callable[[int], Iterator[float]] = default_lambda_scheduler,
        weights: float | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="unet")
        self.student = unet
        self.teacher = copy.deepcopy(unet)
        self.student_temp, self.teacher_temp = temperature
        if isinstance(self.student_temp, float):
            self.student_temp = repeat(self.student_temp)
        if isinstance(self.teacher_temp, float):
            self.teacher_temp = repeat(self.teacher_temp)
        self.lambda_scheduler_fn = lambda_scheduler_fn
        self.weights = weights

        if self.deep_supervision:
            if isinstance(self.weights, tuple):
                assert_eq(len(self.unet.decoder.head), len(self.weights))
            if self.weights is None:
                self.weights = 2.0
            if isinstance(self.weights, float):
                self.weights = tuple(
                    self.weights**i for i in range(len(self.unet.decoder.head))
                )
            self.weights = tensor(self.weights)
            self.weights /= self.weights.sum()
            self.register_buffer("head_weights", self.weights)
        self.automatic_optimization = False

    def training_step(self, batch, _):
        local_views = batch["local"]
        global_views = batch["global"]
        l = next(self.lambda_scheduler)
        st = next(self.student_temp)
        tt = next(self.teacher_temp)

    def configure_optimizers(self):
        self.lambda_scheduler = self.lambda_scheduler_fn(self.trainer.estimated_stepping_batches)
        return super().configure_optimizers()
