import torch

from tessscope.observer.loss import (
    MAXIMUM_INSTANCES,
    ObserverTransform,
    aggregate_depth_loss,
    design_task_loss,
    surrogate_proxy_loss,
)


class TinyObserver(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fcn = torch.nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.pixel_classifier = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )


def test_global_observer_transforms_do_not_use_image_statistics() -> None:
    sensor = torch.tensor([0.0, 1.0, 2.0])
    affine = ObserverTransform("affine", offset=1.0, scale=2.0).apply(sensor)
    transformed = ObserverTransform("asinh", offset=1.0, scale=2.0).apply(sensor)
    assert torch.allclose(affine, torch.tensor([0.0, 0.0, 0.5]))
    assert torch.allclose(
        transformed,
        torch.asinh(torch.tensor([-0.5, 0.0, 0.5])).clamp(0.0, 1.0),
    )


def test_depth_aggregation_matches_frozen_formula() -> None:
    losses = torch.tensor([0.2, 0.4, 0.8])
    temperature = 0.1
    expected = 0.75 * losses.mean() + 0.25 * temperature * torch.log(
        torch.exp(losses / temperature).mean()
    )
    assert torch.allclose(aggregate_depth_loss(losses), expected)


def test_design_loss_has_sensor_and_phase_gradients() -> None:
    torch.manual_seed(3)
    model = TinyObserver().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sensor = torch.rand((1, 3, 96, 96), requires_grad=True)
    phase = torch.linspace(-0.3, 0.3, 6, requires_grad=True)
    labels = torch.zeros((1, 96, 96), dtype=torch.long)
    labels[:, 38:58, 40:56] = 1
    centers = torch.full((1, MAXIMUM_INSTANCES, 2), -1.0)
    centers[:, 0] = torch.tensor([47.5, 47.5])
    valid = torch.zeros((1, MAXIMUM_INSTANCES), dtype=torch.bool)
    valid[:, 0] = True

    loss = design_task_loss(model, sensor, phase, labels, centers, valid)
    sensor_gradient, phase_gradient = torch.autograd.grad(loss, (sensor, phase))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(sensor_gradient).all()
    assert torch.isfinite(phase_gradient).all()
    assert sensor_gradient.abs().sum() > 0
    assert phase_gradient.abs().sum() > 0


def test_surrogate_proxy_has_finite_sensor_gradient() -> None:
    sensor = torch.rand((2, 3, 32, 32), requires_grad=True)
    labels = torch.zeros((2, 32, 32), dtype=torch.long)
    labels[:, 8:24, 10:22] = 1
    loss = surrogate_proxy_loss(sensor, labels)
    (gradient,) = torch.autograd.grad(loss, sensor)
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
