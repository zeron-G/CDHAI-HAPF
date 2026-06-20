import torch

from hapf.models.adapters import FactorizedPatientVariableAdapter


def test_zero_patient_code_is_population_fallback() -> None:
    adapter = FactorizedPatientVariableAdapter(hidden_dim=16, patient_code_dim=4, rank=3)
    with torch.no_grad():
        adapter.up.weight.normal_()
    hidden = torch.randn(5, 16)
    code = torch.zeros(5, 4)
    variable = torch.zeros(5, dtype=torch.long)
    output = adapter(hidden, code, variable)
    assert torch.equal(output, hidden)


def test_nonzero_code_can_change_hidden_state() -> None:
    adapter = FactorizedPatientVariableAdapter(hidden_dim=16, patient_code_dim=4, rank=3)
    with torch.no_grad():
        adapter.up.weight.normal_()
    hidden = torch.randn(5, 16)
    code = torch.ones(5, 4)
    variable = torch.zeros(5, dtype=torch.long)
    output = adapter(hidden, code, variable)
    assert not torch.equal(output, hidden)

