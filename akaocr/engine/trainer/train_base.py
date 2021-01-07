# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon January 1 17:06:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain train method
_____________________________________________________________________________
"""
import os

from engine.solver import ModelCheckpointer, PeriodicCheckpointer
from engine.solver import build_lr_scheduler, build_optimizer
from engine.build import build_dataloader
from engine.build import build_dataloader
from utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)


from utils.utility import initial_logger
logger = initial_logger()


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = ModelCheckpointer(
        model, cfg.SOLVER.EXP, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.SOLVER.WEIGHT, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.SOLVER.EXP, "metrics.json")),
            TensorboardXWriter(cfg.SOLVER.EXP),
        ]
    )

    data_loader = build_dataloader(cfg.SOLVER.DATA_SOURCE, cfg, vocab=cfg.MODEL.VOCAB)

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            print(iteration)

