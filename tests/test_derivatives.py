import numpy as np

from tessscope.validation.derivatives import directional_derivative_report


def test_directional_derivative_report_passes_exact_quadratic() -> None:
    point = np.asarray([0.2, -0.4, 0.7], dtype=np.float32)

    def objective(value: np.ndarray) -> float:
        return float(np.sum(np.asarray(value, dtype=np.float64) ** 2))

    report = directional_derivative_report(objective, point, 2.0 * point)
    assert report["passed"] is True
    assert len(report["stable_epsilons"]) == 3
