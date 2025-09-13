# src/train.py
"""Model and training-related utilities.
In this project we do *not* train models from scratch – we only load
public checkpoints and (optionally) wrap the UNet with the STaR-Diffusion
scheduling engine.  All helper logic that is specific to model loading or
modification lives in this file so that the rest of the code base stays
cleanly separated from model internals.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline

from .config import ModelCfg, HardwareCfg

# ---------------------------------------------------------------------
# Helper – robust pipeline loader
# ---------------------------------------------------------------------


def _load_pipeline(model_cfg: ModelCfg, hw_cfg: HardwareCfg) -> StableDiffusionPipeline:  # noqa: N802 – keep original name
    """Download / cache a pretrained Stable-Diffusion pipeline.

    Parameters
    ----------
    model_cfg : ModelCfg
        Dataclass entry that specifies *what* model has to be loaded.
    hw_cfg : HardwareCfg
        Describes *where* the model should be placed (dtype / device).

    Returns
    -------
    StableDiffusionPipeline
        Ready-to-use diffusers pipeline that has its safety checker
        disabled for maximum throughput.
    """
    # Choose data-type and device **before** any heavy allocations happen
    dtype = getattr(torch, hw_cfg.dtype)
    device = torch.device(hw_cfg.device)

    # Token is optional – only required for gated models.
    # The environment variable name is mandated by the task description.
    token: Optional[str] = os.getenv("HF_TOKEN")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_cfg.repo_id,
        torch_dtype=dtype,
        use_auth_token=token,
    )

    # Push every sub-module to the correct compute device.  We purposefully
    # do *not* wrap this in a try/except; fail-loud is preferable here.
    pipe.to(device)

    # Disable NSFW safety checker to avoid extra compute during benchmarks
    pipe.safety_checker = None

    return pipe
