import pytest

from src.train import select_best_result


def test_select_best_result_uses_highest_f1():
    results = [
        {"run_name": "MLP-01", "f1": 0.81},
        {"run_name": "MLP-02", "f1": 0.89},
    ]

    assert select_best_result(results)["run_name"] == "MLP-02"


def test_select_best_result_rejects_empty_results():
    with pytest.raises(ValueError, match="Nenhum resultado"):
        select_best_result([])
