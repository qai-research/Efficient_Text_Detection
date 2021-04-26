# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Tue January 5 14:01:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain learning rate schedulers class
_____________________________________________________________________________
"""
import math
from bisect import bisect_right
import itertools
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
       
    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

class WarmupDecayCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        milestones: List[int],
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.max_iters = max_iters
        self.milestones = milestones
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        
        min_lr = 10**-5
        self.warmup_factor = 0.001
        indx = bisect_right(self.milestones, self.last_epoch)
        if indx == 0:
            cur_iter = self.last_epoch
            iter_range = self.milestones[0]
            alpha = 1.0
        elif indx == len(self.milestones):
            cur_iter = self.last_epoch - self.milestones[indx-1]
            iter_range = self.max_iters - self.milestones[indx-1]
            alpha = 0.8**indx
        else:
            cur_iter = self.last_epoch - self.milestones[indx-1]
            iter_range = self.milestones[indx] - self.milestones[indx-1]
            alpha = 0.8**indx

        factor = warmup_linear_factor(self.warmup_iters, cur_iter, self.warmup_factor)
        return [base_lr*factor*alpha*decay_cosine_lr(iter_range, self.warmup_iters, cur_iter)+ min_lr 
                for base_lr in self.base_lrs ]
        
    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

def warmup_linear_factor(warmup_iters, iteration, warmup_factor):
    if iteration < warmup_iters:
        alpha = iteration / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        return 1.0
    
def decay_cosine_lr(iter_range, warmup_iters, step):
    if step <= warmup_iters:
        return 1.0
    else:
        return math.cos(math.pi*step/(2*iter_range))