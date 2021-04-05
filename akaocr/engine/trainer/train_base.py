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
import torch.optim as optim

# from engine.metric.evaluation import Evaluation

class Trainer:
    def __init__(self, cfg, model, train_loader=None, test_loader=None, custom_loop=None, accuracy=None, evaluation=None, resume=False):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.custom_loop = custom_loop
        self.accuracy = accuracy
        self.evaluation = evaluation
        self.resume = resume
        self.metric = None

        if test_loader is None:
            logger.warning(f"Validation data not found, training without checkpoint validation")

    def do_test(self, model, data, metric):
        return self.evaluation.run(model, data, metric=metric)

    def do_train(self):
        self.model.train()
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        checkpointer = ModelCheckpointer(
            self.model, self.cfg.SOLVER.EXP, optimizer=optimizer, scheduler=scheduler
        )
        self.cfg.SOLVER.START_ITER = (
                checkpointer.resume_or_load(self.cfg.SOLVER.WEIGHT, resume=self.resume).get("iteration", -1) + 1
        )
        
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=self.cfg.SOLVER.MAX_ITER
        )

        writers = (
            [
                CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
                JSONWriter(os.path.join(self.cfg.SOLVER.EXP, "metrics.json")),
                TensorboardXWriter(self.cfg.SOLVER.EXP),
            ]
        )
       
        with EventStorage(self.cfg.SOLVER.START_ITER) as storage:
            for data, iteration in zip(self.train_loader, range(self.cfg.SOLVER.START_ITER, self.cfg.SOLVER.MAX_ITER)):
                storage.iter = iteration
                loss, acc = self.custom_loop.loop(self.model, data, self.accuracy)
                # print(loss, acc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                storage.put_scalar("loss", loss, smoothing_hint=False)
                if acc != None:
                    storage.put_scalar("acc", acc, smoothing_hint=False)
                scheduler.step()
                writers[2].write()
                if (
                        (iteration) % self.cfg.SOLVER.EVAL_PERIOD == 0
                        and iteration != self.cfg.SOLVER.MAX_ITER - 1
                ):
                    periodic_checkpointer.step(iteration)
                    self.metric, mess = self.do_test(self.model, self.test_loader, self.metric)
                    logger.info(mess)