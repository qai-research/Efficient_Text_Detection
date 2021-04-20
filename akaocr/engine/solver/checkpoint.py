# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Wed January 6 10:38:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file borrow checkpoint loader process from detectron2
_____________________________________________________________________________
"""

import os
import copy
import logging
import re
import torch
from fvcore.common.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
import torch.nn as nn
from utils.torchutils import copy_state_dict
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])

class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


class ModelCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=True, **checkpointables):
        # is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=save_to_disk,
            **checkpointables,
        )
        # if hasattr(self, "path_manager"):
        #     self.path_manager = PathManager
        # else:
        # This could only happen for open source
        # TODO remove after upgrading fvcore version
        # from iopath.common.file_io import g_pathmgr
        #
        # for handler in PathManager._path_handlers.values():
        #     try:
        #         g_pathmgr.register_handler(handler)
        #     except KeyError:
        #         pass

    # def _load_file(self, filename):
    #     if filename.endswith(".pkl"):
    #         with PathManager.open(filename, "rb") as f:
    #             data = pickle.load(f, encoding="latin1")
    #         if "model" in data and "__author__" in data:
    #             # file is in Detectron2 model zoo format
    #             self.logger.info("Reading a file from '{}'".format(data["__author__"]))
    #             return data
    #         else:
    #             # assume file is from Caffe2 / Detectron1 model zoo
    #             if "blobs" in data:
    #                 # Detection models have "blobs", but ImageNet models don't
    #                 data = data["blobs"]
    #             data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
    #             return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
    #     loaded = super()._load_file(filename)  # load native pth checkpoint
    #     if "model" not in loaded:
    #         loaded = {"model": loaded}
    #     return loaded

    # def _load_model(self, checkpoint):
    #     if checkpoint.get("matching_heuristics", False):
    #         self._convert_ndarray_to_tensor(checkpoint["model"])
    #         # convert weights by name-matching heuristics
    #         model_state_dict = self.model.state_dict()
    #         align_and_update_state_dicts(
    #             model_state_dict,
    #             checkpoint["model"],
    #             c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
    #         )
    #         checkpoint["model"] = model_state_dict
    #     # for non-caffe2 models, use standard ways to load it
    #     incompatible = super()._load_model(checkpoint)
    #     if incompatible is None:  # support older versions of fvcore
    #         return None

    #     model_buffers = dict(self.model.named_buffers(recurse=False))
    #     for k in ["pixel_mean", "pixel_std"]:
    #         # Ignore missing key message about pixel_mean/std.
    #         # Though they may be missing in old checkpoints, they will be correctly
    #         # initialized from config anyway.
    #         if k in model_buffers:
    #             try:
    #                 incompatible.missing_keys.remove(k)
    #             except ValueError:
    #                 pass
    #     return incompatible
    def resume_or_load(self, path: str, *, resume: bool = True, strict_mode) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path, strict_mode=strict_mode)
        else:
            return self.load(path, strict_mode=strict_mode, checkpointables=[])


    def _load_file(self, f: str) -> Dict[str, Any]:
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.

        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return copy_state_dict(torch.load(f, map_location=torch.device("cpu")))

    def _load_model(self, checkpoint: Any, strict_mode) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        # print(checkpoint)
        checkpoint_state_dict = checkpoint
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # workaround https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                # Allow mismatch for uninitialized parameters
                if TORCH_VERSION >= (1, 8) and isinstance(
                    model_param, nn.parameter.UninitializedParameter
                ):
                    continue
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:

                    has_observer_base_classes = (
                        TORCH_VERSION >= (1, 8)
                        and hasattr(torch.quantization, "ObserverBase")
                        and hasattr(torch.quantization, "FakeQuantizeBase")
                    )
                    if has_observer_base_classes:
                        # Handle the special case of quantization per channel observers,
                        # where buffer shape mismatches are expected.
                        def _get_module_for_key(
                            model: torch.nn.Module, key: str
                        ) -> torch.nn.Module:
                            # foo.bar.param_or_buffer_name -> [foo, bar]
                            key_parts = key.split(".")[:-1]
                            cur_module = model
                            for key_part in key_parts:
                                cur_module = getattr(cur_module, key_part)
                            return cur_module

                        cls_to_skip = (
                            torch.quantization.ObserverBase,
                            torch.quantization.FakeQuantizeBase,
                        )
                        target_module = _get_module_for_key(self.model, k)
                        if isinstance(target_module, cls_to_skip):
                            # Do not remove modules with expected shape mismatches
                            # them from the state_dict loading. They have special logic
                            # in _load_from_state_dict to handle the mismatches.
                            continue

                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=strict_mode)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )


    def load(
        self, path: str, strict_mode, checkpointables: Optional[List[str]] = None, 
    ) -> Dict[str, Any]:
        """
        Load from the given checkpoint.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint, strict_mode=strict_mode)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint

    
def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)



# Note the current matching is not symmetric.
# it assumes model_state_dict will have longer names.
def align_and_update_state_dicts(model_state_dict, ckpt_state_dict, c2_conversion=False):
    """
    Match names between the two state-dict, and update the values of model_state_dict in-place with
    copies of the matched tensor in ckpt_state_dict.
    If `c2_conversion==True`, `ckpt_state_dict` is assumed to be a Caffe2
    model and will be renamed at first.

    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(model_state_dict.keys())
    # if c2_conversion:
    #     ckpt_state_dict, original_keys = convert_c2_detectron_names(ckpt_state_dict)
    #     # original_keys: the name in the original dict (before renaming)
    # else:
    #     original_keys = {x: x for x in ckpt_state_dict.keys()}
    original_keys = {x: x for x in ckpt_state_dict.keys()}
    ckpt_keys = sorted(ckpt_state_dict.keys())

    def match(a, b):
        # Matched ckpt_key should be a complete (starts with '.') suffix.
        # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
        # but matches whatever_conv1 or mesh_head.whatever_conv1.
        return a == b or a.endswith("." + b)

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # ckpt_key string, if it matches
    match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
    # use the matched one with longest size in case of multiple matches
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_len_model = max(len(key) for key in model_keys) if model_keys else 1
    max_len_ckpt = max(len(key) for key in ckpt_keys) if ckpt_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    # matched_pairs (matched checkpoint key --> matched model key)
    matched_keys = {}
    for idx_model, idx_ckpt in enumerate(idxs.tolist()):
        if idx_ckpt == -1:
            continue
        key_model = model_keys[idx_model]
        key_ckpt = ckpt_keys[idx_ckpt]
        value_ckpt = ckpt_state_dict[key_ckpt]
        shape_in_model = model_state_dict[key_model].shape

        if shape_in_model != value_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_ckpt, value_ckpt.shape, key_model, shape_in_model
                )
            )
            logger.warning(
                "{} will not be loaded. Please double check and see if this is desired.".format(
                    key_ckpt
                )
            )
            continue

        model_state_dict[key_model] = value_ckpt.clone()
        if key_ckpt in matched_keys:  # already added to matched_keys
            logger.error(
                "Ambiguity found for {} in checkpoint!"
                "It matches at least two keys in the model ({} and {}).".format(
                    key_ckpt, key_model, matched_keys[key_ckpt]
                )
            )
            raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

        matched_keys[key_ckpt] = key_model
        logger.info(
            log_str_template.format(
                key_model,
                max_len_model,
                original_keys[key_ckpt],
                max_len_ckpt,
                tuple(shape_in_model),
            )
        )
    matched_model_keys = matched_keys.values()
    matched_ckpt_keys = matched_keys.keys()
    # print warnings about unmatched keys on both side
    unmatched_model_keys = [k for k in model_keys if k not in matched_model_keys]
    if len(unmatched_model_keys):
        logger.info(get_missing_parameters_message(unmatched_model_keys))

    unmatched_ckpt_keys = [k for k in ckpt_keys if k not in matched_ckpt_keys]
    if len(unmatched_ckpt_keys):
        logger.info(
            get_unexpected_parameters_message(original_keys[x] for x in unmatched_ckpt_keys)
        )
