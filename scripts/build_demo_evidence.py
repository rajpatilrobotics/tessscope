"""Build a compact, deterministic validation-only replay evidence pack."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from tessscope.demo.evidence import (
    EXACT_NAME,
    PIECEWISE_NAME,
    STOPPED_NAME,
    sha256_array,
    sha256_path,
    write_deterministic_npz,
)
from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import segment_sensor_batch
from tessscope.observer.instanseg import load_frozen_model
from tessscope.observer.loss import ObserverTransform
from tessscope.v2.autofocus.features import spectral_features
from tessscope.v2.autofocus.ridge import ridge_predict
from tessscope.v2.data.bbbc006 import records_for_split
from tessscope.v2.optimization.served import DEPTHS, V2SystemCalibration, collect_patches
from tessscope.v2_1.optics.model import (
    phase_coefficients_b7,
    psf_sensor_b7,
    simulate_noisy_sensor_b7,
    zernike_basis_b7,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "demo" / "representative-selection.json"
)
V2_4_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
PIECEWISE_HARD_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "validation"
    / "piecewise-hard-frontier.json"
)
PIECEWISE_DESIGN_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "screening"
    / "piecewise-soft-frontier.json"
)
EXACT_DESIGN_PATH = PROJECT_ROOT / "configs" / "v2_4" / "selected-checkpoints.json"
STOPPED_DESIGN_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "stopped-stage-matched.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "demo"
ARRAYS_PATH = OUTPUT_ROOT / "validation-replay.npz"
METADATA_PATH = OUTPUT_ROOT / "validation-replay.json"
SYSTEMS = ("clear", "exact", "stopped", "piecewise")
CORRECTED_SYSTEMS = ("exact", "stopped", "piecewise")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _parameters() -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    exact_source = _json(EXACT_DESIGN_PATH)
    exact_index, exact = next(
        (index, row)
        for index, row in enumerate(exact_source["selected"])
        if row["name"] == EXACT_NAME
    )
    stopped_source = _json(STOPPED_DESIGN_PATH)
    pair_index, stopped_pair = next(
        (index, row)
        for index, row in enumerate(stopped_source["pairs"])
        if row["stopped"]["name"] == STOPPED_NAME
    )
    piecewise_source = _json(PIECEWISE_DESIGN_PATH)
    piecewise_index, piecewise = next(
        (index, row)
        for index, row in enumerate(piecewise_source["points"])
        if row["name"] == PIECEWISE_NAME
    )
    parameters = {
        "clear": np.zeros(7, dtype=np.float32),
        "exact": np.asarray(exact["parameters"], dtype=np.float32),
        "stopped": np.asarray(stopped_pair["stopped"]["final_parameters"], dtype=np.float32),
        "piecewise": np.asarray(piecewise["parameters"], dtype=np.float32),
    }
    trace = {
        "clear": {
            "name": "clear",
            "source": "analytic zero B7 pupil",
            "pointer": None,
        },
        "exact": {
            "name": EXACT_NAME,
            "source": str(EXACT_DESIGN_PATH.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_path(EXACT_DESIGN_PATH),
            "pointer": f"/selected/{exact_index}/parameters",
            "parameter_sha256": exact["parameter_sha256"],
        },
        "stopped": {
            "name": STOPPED_NAME,
            "source": str(STOPPED_DESIGN_PATH.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_path(STOPPED_DESIGN_PATH),
            "pointer": f"/pairs/{pair_index}/stopped/final_parameters",
        },
        "piecewise": {
            "name": PIECEWISE_NAME,
            "source": str(PIECEWISE_DESIGN_PATH.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_path(PIECEWISE_DESIGN_PATH),
            "pointer": f"/points/{piecewise_index}/parameters",
        },
    }
    return parameters, trace


def _simulate(
    parameters: np.ndarray,
    objects: np.ndarray,
    depths: np.ndarray,
    calibration: V2SystemCalibration,
) -> np.ndarray:
    noise = np.zeros((len(objects), len(depths), 256, 256), dtype=np.float32)
    result = simulate_noisy_sensor_b7(
        jnp.asarray(parameters),
        jnp.asarray(objects),
        jnp.asarray(depths),
        jnp.asarray(noise),
        expected_photons=calibration.expected_photons,
        exposure_gain=calibration.exposure_gain,
        depth_scale=calibration.depth_scale,
        axial_offset_um=calibration.axial_offset_um,
    )
    return np.maximum(np.asarray(result), 0.0).astype(np.float32)


def _row_index(rows: list[dict], design: str, field_id: str, depth: float, key: str) -> int:
    matches = [
        index
        for index, row in enumerate(rows)
        if row["design"] == design
        and row["field_id"] == field_id
        and float(row[key]) == depth
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {design} {field_id} {depth} row, got {matches}")
    return matches[0]


def _pq(target: np.ndarray, prediction: np.ndarray) -> float:
    valid_target, valid_prediction = valid_region_labels(target, prediction, margin=62)
    return instance_metrics(valid_target, valid_prediction).panoptic_quality


def _phase_and_psf(parameters: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = jnp.linspace(-1.0, 1.0, 256, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    grid = jnp.stack([grid_y, grid_x], axis=-1)
    mask = np.asarray(grid_x * grid_x + grid_y * grid_y <= 1.0, dtype=np.uint8)
    phases = []
    psfs = []
    for name in SYSTEMS:
        value = jnp.asarray(parameters[name])
        coefficients = phase_coefficients_b7(value)
        phase = jnp.einsum("m,mhw->hw", coefficients, zernike_basis_b7(grid, 1.0))
        phases.append(np.asarray(phase, dtype=np.float32))
        depth_psfs = jax.vmap(lambda depth, value=value: psf_sensor_b7(value, depth))(
            jnp.asarray(DEPTHS)
        )
        psfs.append(np.asarray(depth_psfs, dtype=np.float32))
    return np.stack(phases), np.stack(psfs), mask


def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, dict]:
    return {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": sha256_array(array),
        }
        for name, array in sorted(arrays.items())
    }


def main() -> None:
    selection = _json(SELECTION_PATH)
    if selection["split"] != "validation" or selection["test_accessed"]:
        raise ValueError("Representative selection is not validation-only and sealed")
    field_id = selection["selected_patch"]["field_id"]

    patches = collect_patches("validation", 45, seed=53)
    query_index = next(index for index, patch in enumerate(patches) if patch.field_id == field_id)
    query = patches[query_index]
    support = [
        patches[(query_index + 1) % len(patches)],
        patches[(query_index + 2) % len(patches)],
    ]
    if list(query.origin_yx) != selection["selected_patch"]["origin_yx"]:
        raise ValueError("Selected validation crop origin changed")
    objects = np.stack([support[0].object_image, support[1].object_image, query.object_image])

    calibration = V2SystemCalibration.load()
    parameters, design_trace = _parameters()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = ObserverTransform(mode="affine", offset=0.0, scale=calibration.exposure_gain)
    hard_v2_4 = _json(V2_4_PATH)
    hard_piecewise = _json(PIECEWISE_HARD_PATH)

    sensors_before = []
    labels_before = []
    sensors_corrected = []
    labels_corrected = []
    predicted_depths = []
    residual_depths = []
    frames = []

    for system in SYSTEMS:
        sensor = _simulate(parameters[system], objects, DEPTHS, calibration)
        features, _ = spectral_features(sensor.reshape(-1, 256, 256))
        ridge = ridge_predict(
            features[:14],
            np.tile(DEPTHS, 2),
            features[14:],
            ridge_lambda=calibration.ridge_lambda,
        )
        prediction = segment_sensor_batch(
            model,
            sensor[2],
            query.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        residual = DEPTHS + np.clip(-ridge.predictions, -6.0, 6.0)
        sensors_before.append(sensor[2])
        labels_before.append(prediction)
        predicted_depths.append(np.asarray(ridge.predictions, dtype=np.float64))
        residual_depths.append(np.asarray(residual, dtype=np.float64))

        if system == "clear":
            continue
        corrected = _simulate(
            parameters[system], query.object_image[None], residual.astype(np.float32), calibration
        )[0]
        corrected_prediction = segment_sensor_batch(
            model,
            corrected,
            query.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        sensors_corrected.append(corrected)
        labels_corrected.append(corrected_prediction)

        artifact = hard_piecewise if system == "piecewise" else hard_v2_4
        design = design_trace[system]["name"]
        source_path = PIECEWISE_HARD_PATH if system == "piecewise" else V2_4_PATH
        for depth_index, depth_value in enumerate(DEPTHS):
            depth = float(depth_value)
            before_index = _row_index(artifact["rows"], design, field_id, depth, "depth_um")
            before_row = artifact["rows"][before_index]
            observed_before = _pq(query.instance_labels, prediction[depth_index])
            if observed_before != before_row["panoptic_quality"]:
                raise ValueError(f"InstanSeg before-PQ mismatch for {system} at {depth:+.1f} µm")
            if not np.isclose(ridge.predictions[depth_index], before_row["predicted_depth_um"]):
                raise ValueError(f"Autofocus prediction mismatch for {system} at {depth:+.1f} µm")
            frame = {
                "system": system,
                "design": design,
                "depth_index": depth_index,
                "depth_um": depth,
                "predicted_depth_um": float(ridge.predictions[depth_index]),
                "residual_depth_um": float(residual[depth_index]),
                "before_pq": float(observed_before),
                "source": str(source_path.relative_to(PROJECT_ROOT)),
                "source_sha256": sha256_path(source_path),
                "before_pointer": f"/rows/{before_index}",
            }
            if depth != 0.0:
                corrected_index = _row_index(
                    artifact["corrected_rows"],
                    design,
                    field_id,
                    depth,
                    "original_depth_um",
                )
                corrected_row = artifact["corrected_rows"][corrected_index]
                observed_after = _pq(query.instance_labels, corrected_prediction[depth_index])
                if observed_after != corrected_row["after_pq"]:
                    raise ValueError(
                        f"InstanSeg corrected-PQ mismatch for {system} at {depth:+.1f} µm"
                    )
                if not np.isclose(residual[depth_index], corrected_row["residual_depth_um"]):
                    raise ValueError(f"Residual-depth mismatch for {system} at {depth:+.1f} µm")
                frame.update(
                    {
                        "after_pq": float(observed_after),
                        "corrected_pointer": f"/corrected_rows/{corrected_index}",
                    }
                )
            frames.append(frame)

    phases, psfs, pupil_mask = _phase_and_psf(parameters)
    arrays = {
        "object_image": query.object_image.astype(np.float32),
        "target_labels": query.instance_labels.astype(np.int32),
        "sensor_before": np.stack(sensors_before).astype(np.float32),
        "labels_before": np.stack(labels_before).astype(np.int32),
        "sensor_corrected": np.stack(sensors_corrected).astype(np.float32),
        "labels_corrected": np.stack(labels_corrected).astype(np.int32),
        "predicted_depth_um": np.stack(predicted_depths).astype(np.float64),
        "residual_depth_um": np.stack(residual_depths).astype(np.float64),
        "phase_parameters": np.stack([parameters[name] for name in SYSTEMS]).astype(np.float32),
        "pupil_phase_radians": phases,
        "pupil_mask": pupil_mask,
        "psf_sensor": psfs,
    }
    write_deterministic_npz(ARRAYS_PATH, arrays)

    records = {record.field_id: record for record in records_for_split("validation")}
    patch_trace = []
    for patch in (*support, query):
        record = records[patch.field_id]
        object_path = record.image_paths[3]
        patch_trace.append(
            {
                "role": "query" if patch.field_id == query.field_id else "autofocus_support",
                "field_id": patch.field_id,
                "well": patch.well,
                "split": patch.split,
                "origin_yx": list(patch.origin_yx),
                "object_source": str(object_path.relative_to(PROJECT_ROOT)),
                "object_source_sha256": sha256_path(object_path),
                "label_source": str(record.label_path.relative_to(PROJECT_ROOT)),
                "label_source_sha256": sha256_path(record.label_path),
            }
        )

    metadata = {
        "schema_version": 1,
        "status": "complete_cached_validation_replay",
        "mode": "cached_replay_not_live_inference",
        "test_accessed": False,
        "split": "validation",
        "selected_field": field_id,
        "selected_display_depth_um": selection["selected_display_depth"]["depth_um"],
        "system_order": list(SYSTEMS),
        "corrected_system_order": list(CORRECTED_SYSTEMS),
        "depths_um": [float(depth) for depth in DEPTHS],
        "observer": {
            "name": "InstanSeg",
            "device_used_to_materialize_cache": str(device),
            "transform": {
                "mode": "affine",
                "offset": 0.0,
                "scale": calibration.exposure_gain,
                "clip": [0.0, 1.0],
                "per_image_normalization": False,
            },
            "valid_metric_margin_px": 62,
        },
        "calibration": {
            "expected_photons": calibration.expected_photons,
            "exposure_gain": calibration.exposure_gain,
            "depth_scale": calibration.depth_scale,
            "axial_offset_um": calibration.axial_offset_um,
            "ridge_lambda": calibration.ridge_lambda,
        },
        "patches": patch_trace,
        "designs": design_trace,
        "frames": frames,
        "arrays": _array_manifest(arrays),
        "archive": {
            "path": str(ARRAYS_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(ARRAYS_PATH),
            "size_bytes": ARRAYS_PATH.stat().st_size,
        },
        "frozen_inputs": {
            "selection": {
                "path": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(SELECTION_PATH),
            },
            "v2_4_hard": {
                "path": str(V2_4_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(V2_4_PATH),
            },
            "piecewise_hard": {
                "path": str(PIECEWISE_HARD_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(PIECEWISE_HARD_PATH),
            },
        },
        "limitations": [
            "Validation-only cached replay; no locked-test evidence.",
            "Reference labels are automated BBBC006/CellProfiler instances, not manual truth.",
            "Sensor frames are deterministic in-silico outputs, not physical microscope captures.",
            "No scale bar is shown because an authoritative display-plane calibration is absent.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"device={device}")
    print(f"arrays={ARRAYS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"arrays_sha256={sha256_path(ARRAYS_PATH)}")
    print(f"arrays_size_bytes={ARRAYS_PATH.stat().st_size}")
    print(f"metadata={METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"metadata_sha256={sha256_path(METADATA_PATH)}")


if __name__ == "__main__":
    main()
