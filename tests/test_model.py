import torch

from hapf.config import ModelConfig
from hapf.models.forecaster import PatientAdaptiveForecaster


def test_forecaster_shapes_and_positive_scale() -> None:
    config = ModelConfig(input_channels=4, hidden_dim=16, levels=2, patient_code_dim=5, adapter_rank=3)
    model = PatientAdaptiveForecaster(config, horizons=2)
    features = torch.randn(7, 4, 24)
    code = torch.randn(7, 5)
    location, scale = model(features, patient_code=code)
    assert location.shape == (7, 2)
    assert scale.shape == (7, 2)
    assert torch.all(scale > 0)


def test_causal_model_does_not_use_future_input_positions() -> None:
    config = ModelConfig(input_channels=4, hidden_dim=16, levels=2, patient_code_dim=5, adapter_rank=3, dropout=0.0)
    model = PatientAdaptiveForecaster(config, horizons=2).eval()
    original = torch.randn(1, 4, 24)
    altered = original.clone()
    altered[..., -1] += 100.0
    first_prefix = model.backbone.blocks[0](model.backbone.input_projection(original))[..., :-1]
    second_prefix = model.backbone.blocks[0](model.backbone.input_projection(altered))[..., :-1]
    assert torch.allclose(first_prefix, second_prefix, atol=1e-6)

