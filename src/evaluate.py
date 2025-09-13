# src/evaluate.py
"""Evaluation, metrics, and high-level experiment driver."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tqdm
from ptflops import get_model_complexity_info
from torchmetrics.image.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance

from .config import DatasetCfg, ExperimentCfg, ModelCfg
from .preprocess import prepare_coco_prompts
from .train import _load_pipeline

# ---------------------------------------------------------------------
# General-purpose utilities (kept local to stay within the 6-file limit)
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Stopwatch:
    """Simple context-manager for wall-clock timings."""

    def __enter__(self):  # noqa: D401 (we want a short name)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001, D401
        self.elapsed = time.perf_counter() - self._start


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------


class MetricBundle:
    """Bundle FID + CLIP for *online* accumulation."""

    def __init__(self, device: torch.device):
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    @torch.no_grad()
    def update(self, images: torch.Tensor, prompts: List[str]):
        # images expected in 0..1 range, float32, shape Bx3xHxW
        self.fid.update(images, real=False)
        self.clip.update(images, prompts)

    def compute(self) -> Dict[str, float]:
        return {
            "fid": float(self.fid.compute()),
            "clip": float(self.clip.compute()),
        }


def compute_gmacs(model: torch.nn.Module, input_res: tuple[int, int, int] = (4, 64, 64)) -> float:
    macs, _params = get_model_complexity_info(
        model, input_res, as_strings=False, print_per_layer_stat=False, verbose=False
    )
    return macs / 1e9


# ---------------------------------------------------------------------
# Visualisation (kept minimal – PDF saved for paper figures)
# ---------------------------------------------------------------------


def fid_vs_gmacs(points: List[Dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless back-end for cluster jobs
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(7, 5))
    for entry in points:
        plt.scatter(entry["gmacs"], entry["fid"], label=entry["label"])
        plt.text(entry["gmacs"], entry["fid"], f"{entry['fid']:.2f}")
    plt.xscale("log")
    plt.xlabel("Compute (GMACs)")
    plt.ylabel("FID ↓")
    plt.title("FID vs. Compute Pareto Frontier")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# High-level experiment loop (speed / quality trade-off)
# ---------------------------------------------------------------------


_IMAGES_DIR = Path(".research/iteration1/images")
_JSON_DIR = Path(".research/iteration1")


def _pil_to_tensor(pil_img):  # local helper – avoids circular imports
    arr = np.array(pil_img).astype(np.float32) / 255.0  # H x W x C, 0..1
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1 x 3 x H x W


def run_experiment(cfg: ExperimentCfg) -> None:  # noqa: C901 – keep monolithic for clarity
    """Main driver: downloads data, runs baselines + STaR, saves JSON + figure."""

    # Prepare prompts (download is cached – safe for repeated calls)
    prompts_file = prepare_coco_prompts(cfg.dataset, Path("data/coco"))
    with open(prompts_file, "r", encoding="utf-8") as handle:
        all_prompts = [l.strip() for l in handle.readlines()]

    # Accumulate results for Pareto figure
    pareto: List[Dict] = []

    for model_cfg in cfg.models:
        for seed in cfg.random_seeds:
            set_seed(seed)

            # ------------------------------------------------------------------
            # 1) Vanilla sampler (baseline)
            # ------------------------------------------------------------------
            pipe = _load_pipeline(model_cfg, cfg.hardware)
            device = next(pipe.unet.parameters()).device  # after .to() call in loader

            mb = MetricBundle(device)
            gmacs_full = compute_gmacs(pipe.unet)
            with Stopwatch() as sw:
                for prompt in tqdm.tqdm(all_prompts, desc=f"{model_cfg.name}/vanilla"):
                    img = pipe(
                        prompt,
                        num_inference_steps=model_cfg.inference_steps,
                        height=cfg.dataset.resolutions[0],
                        width=cfg.dataset.resolutions[0],
                    ).images[0]

                    tensor_img = _pil_to_tensor(img).to(device)
                    mb.update(tensor_img, [prompt])

            quality = mb.compute()
            baseline_res = {
                "model": model_cfg.name,
                "method": "vanilla",
                "seed": seed,
                "fid": quality["fid"],
                "clip": quality["clip"],
                "gmacs": gmacs_full,
                "images_per_s": len(all_prompts) / sw.elapsed,
            }
            pareto.append({"label": f"{model_cfg.name}-vanilla", **baseline_res})
            save_json(baseline_res, _JSON_DIR / f"{model_cfg.name}_vanilla_seed{seed}.json")
            print(json.dumps(baseline_res, indent=2))

            # ------------------------------------------------------------------
            # 2) STaR-Diffusion (requires external lib – we fail loud if absent)
            # ------------------------------------------------------------------
            try:
                from star_diffusion import StarEngine
            except ImportError as err:
                print("[WARNING] STaR-Diffusion library missing – skipping sparse runs.")
                continue  # baseline results are still valuable for a smoke test

            for eps in model_cfg.star_eps:
                mb = MetricBundle(device)
                engine = StarEngine(pipe.unet, budget_eps=eps, safety_delta=0.6)
                pipe.unet = engine.wrapped_unet  # hot-swap UNet with sparse wrapper
                gmacs_sparse = engine.estimated_gmacs()

                with Stopwatch() as sw:
                    for prompt in tqdm.tqdm(all_prompts, desc=f"{model_cfg.name}/star-{eps}"):
                        img = pipe(
                            prompt,
                            num_inference_steps=model_cfg.inference_steps,
                            height=cfg.dataset.resolutions[0],
                            width=cfg.dataset.resolutions[0],
                        ).images[0]
                        tensor_img = _pil_to_tensor(img).to(device)
                        mb.update(tensor_img, [prompt])

                quality = mb.compute()
                star_res = {
                    "model": model_cfg.name,
                    "method": "star",
                    "eps": eps,
                    "seed": seed,
                    "fid": quality["fid"],
                    "clip": quality["clip"],
                    "gmacs": gmacs_sparse,
                    "images_per_s": len(all_prompts) / sw.elapsed,
                }
                pareto.append({"label": f"{model_cfg.name}-star-{eps}", **star_res})
                save_json(star_res, _JSON_DIR / f"{model_cfg.name}_star_eps{eps}_seed{seed}.json")
                print(json.dumps(star_res, indent=2))

    # ----------------------------------------------------------------------
    # Global figure – one PDF per *experiment* (smoke / full)
    # ----------------------------------------------------------------------
    fid_vs_gmacs(pareto, _JSON_DIR / "fid_vs_gmacs.pdf")
    print("Saved Pareto figure →", (_JSON_DIR / "fid_vs_gmacs.pdf").resolve())
