"""Gate 0 release parity and Apple-silicon benchmark."""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from tessscope.observer.instanseg import (
    MODEL_BUNDLE_DIR,
    InstanSegSemantics,
    canonicalize_labels,
    load_frozen_model,
    normalize_release_input,
    raw_head,
    reconstruct_hard_labels,
)


def _object_count(labels: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(labels)))


def _foreground_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left > 0, right > 0).sum()
    union = np.logical_or(left > 0, right > 0).sum()
    return float(intersection / union) if union else 1.0


def run_parity(model_dir: Path = MODEL_BUNDLE_DIR) -> dict[str, object]:
    """Return locked release and raw-head reconstruction parity evidence."""
    raw_input = np.load(model_dir / "test-input.npy")
    expected = np.load(model_dir / "test-output_instance_segmentation.npy")
    normalized = normalize_release_input(raw_input)
    model = load_frozen_model("cpu", model_dir / "instanseg.pt")
    semantics = InstanSegSemantics.from_model(model)

    with torch.inference_mode():
        sensor = torch.from_numpy(normalized.copy())
        official = model(sensor.clone()).cpu().numpy()
        prediction = raw_head(model, sensor)
        reconstructed = reconstruct_hard_labels(model, prediction).cpu().numpy()

    expected_canonical = canonicalize_labels(expected)
    reconstructed_canonical = canonicalize_labels(reconstructed)
    report: dict[str, object] = {
        "official_pixel_exact": bool(np.array_equal(official, expected)),
        "reconstructed_pixel_exact_after_deterministic_relabel": bool(
            np.array_equal(reconstructed_canonical, expected_canonical)
        ),
        "reconstructed_different_pixels_after_relabel": int(
            np.count_nonzero(reconstructed_canonical != expected_canonical)
        ),
        "expected_object_count": _object_count(expected),
        "official_object_count": _object_count(official),
        "reconstructed_object_count": _object_count(reconstructed),
        "reconstructed_foreground_iou": _foreground_iou(reconstructed, expected),
        "raw_head_shape": list(prediction.shape),
        "semantics": {
            **semantics.__dict__,
            "channel_order": [
                "coord_x_logit",
                "coord_y_logit",
                "sigma_0",
                "sigma_1",
                "seed_distance_logit",
            ],
            "spatial_embedding": "(sigmoid(coords)-0.5)*8 + linear coordinate map",
            "seed_probability_proxy": "seed_distance_logit/15 + 0.5",
        },
    }
    report["passed"] = bool(
        report["official_pixel_exact"]
        and report["reconstructed_pixel_exact_after_deterministic_relabel"]
        and report["expected_object_count"] == report["reconstructed_object_count"]
    )
    return report


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _time_ms(operation: Callable[[], object], device: torch.device, repeats: int) -> list[float]:
    values: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        operation()
        _synchronize(device)
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def benchmark_device(device_name: str, repeats: int = 5) -> dict[str, object]:
    """Benchmark raw forward, actual input VJP, and hard postprocessing."""
    device = torch.device(device_name)
    raw_input = np.load(MODEL_BUNDLE_DIR / "test-input.npy")
    normalized = normalize_release_input(raw_input)
    model = load_frozen_model(device)
    base = torch.from_numpy(normalized.copy()).to(device)

    with torch.inference_mode():
        for _ in range(2):
            raw_head(model, base)
        cached_head = raw_head(model, base)

    forward_times = _time_ms(
        lambda: raw_head(model, base),
        device,
        repeats,
    )

    def run_vjp() -> None:
        sensor = base.detach().clone().requires_grad_(True)
        prediction = raw_head(model, sensor)
        scalar = prediction.square().mean()
        (gradient,) = torch.autograd.grad(scalar, sensor)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite InstanSeg input VJP")

    run_vjp()
    vjp_times = _time_ms(run_vjp, device, repeats)

    def run_official_hard() -> None:
        with torch.inference_mode():
            model(base.clone())

    run_official_hard()
    official_hard_times = _time_ms(run_official_hard, device, repeats)
    result: dict[str, object] = {
        "device": device_name,
        "repeats": repeats,
        "raw_head_forward": _summary(forward_times),
        "raw_head_forward_and_vjp": _summary(vjp_times),
        "official_hard_end_to_end": _summary(official_hard_times),
        "estimated_hard_postprocess_overhead_ms": max(
            0.0,
            statistics.median(official_hard_times) - statistics.median(forward_times),
        ),
    }
    if device.type == "cpu":
        def run_source_hard() -> None:
            with torch.inference_mode():
                reconstruct_hard_labels(model, cached_head)

        run_source_hard()
        source_hard_times = _time_ms(run_source_hard, device, repeats)
        result["source_hard_postprocess_from_cached_head"] = _summary(source_hard_times)
    else:
        result["source_hard_postprocess_from_cached_head"] = {
            "status": "cpu_only_for_parity",
            "reason": (
                "Upstream eager source helper mixes CPU and MPS tensors in torch.isin; "
                "the bundled official hard endpoint contains its own working MPS path."
            ),
        }
    return result


def run_benchmarks(repeats: int = 5) -> dict[str, object]:
    """Benchmark CPU and MPS when available."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    results: dict[str, object] = {
        "host": platform.platform(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "devices": {},
    }
    for device in devices:
        try:
            results["devices"][device] = benchmark_device(device, repeats=repeats)  # type: ignore[index]
        except Exception as error:  # preserve evidence instead of hiding device failures
            results["devices"][device] = {  # type: ignore[index]
                "device": device,
                "error": f"{type(error).__name__}: {error}",
            }
    return results


def main() -> None:
    """Write machine-readable Gate 0 evidence and fail if parity is not exact."""
    output_dir = Path("artifacts/runs/gate0")
    output_dir.mkdir(parents=True, exist_ok=True)
    parity = run_parity()
    (output_dir / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(json.dumps(parity, indent=2))
    if not parity["passed"]:
        raise SystemExit("Gate 0 parity failed; do not continue to data plumbing")

    benchmarks = run_benchmarks()
    (output_dir / "benchmark.json").write_text(json.dumps(benchmarks, indent=2) + "\n")
    print(json.dumps(benchmarks, indent=2))


if __name__ == "__main__":
    main()
