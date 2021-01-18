# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon January 01 17:06:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain training procedure
_____________________________________________________________________________
"""
import os

from engine.solver import ModelCheckpointer, PeriodicCheckpointer
from engine.solver import build_lr_scheduler, build_optimizer
from engine.build import build_dataloader
from utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from utils.utility import initial_logger

logger = initial_logger()


def do_train(cfg, model, custom_loop=None, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = ModelCheckpointer(
        model, cfg.SOLVER.EXP, optimizer=optimizer, scheduler=scheduler
    )
    cfg.SOLVER.START_ITER = (
            checkpointer.resume_or_load(cfg.SOLVER.WEIGHT, resume=resume).get("iteration", -1) + 1
    )

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=cfg.SOLVER.MAX_ITER
    )

    writers = (
        [
            CommonMetricPrinter(cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(cfg.SOLVER.EXP, "metrics.json")),
            TensorboardXWriter(cfg.SOLVER.EXP),
        ]
    )

    data_loader = build_dataloader(cfg)

    with EventStorage(cfg.SOLVER.START_ITER) as storage:
        for data, iteration in zip(data_loader, range(cfg.SOLVER.START_ITER, cfg.SOLVER.MAX_ITER)):
            storage.iter = iteration
            loss = custom_loop.loop(model, data)
            print(iteration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            





