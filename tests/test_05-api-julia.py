from pipelines.expansion.api import ExpansionModel


def test_expansion_model():
    expansion_model = ExpansionModel()
    results = expansion_model.run_sddp()
    assert results is not None
    assert results.status_code == 200
