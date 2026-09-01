import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_locked_test.py"
SPEC = importlib.util.spec_from_file_location("evaluate_locked_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_report = MODULE.build_report


def _row(source: str, design: str, depth: float, pq: float, count_error: int = 2):
    return {
        "source_image_id": source,
        "design": design,
        "kind": "deterministic",
        "depth_um": depth,
        "expected_photons": 100,
        "replicate": -1,
        "panoptic_quality": pq,
        "segmentation_quality": pq,
        "recognition_quality": pq,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "predicted_count": 1,
        "target_count": 1,
        "absolute_count_error": count_error,
        "percentage_count_error": float(count_error),
        "foreground_dice": pq,
    }


def test_report_keeps_negative_result_when_thresholds_fail() -> None:
    rows = []
    designs = ("clear", "cubic", "image_fidelity", "exact_task", "surrogate")
    for source in ("a", "b"):
        for design in designs:
            for depth in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0):
                rows.append(_row(source, design, depth, 0.5))
            for photon_level in (50, 200):
                for depth in (-6.0, 6.0):
                    for replicate in range(4):
                        row = _row(source, design, depth, 0.5)
                        row.update(
                            kind="poisson",
                            expected_photons=photon_level,
                            replicate=replicate,
                        )
                        rows.append(row)
    report = build_report(rows, 1.0)
    assert report["evaluation_completed"]
    assert not report["positive_headline_claim"]
