import pytest
import torch

from flare_nrt.model import Ensemble, WeightsError, clean_keys, sha256


def test_clean_keys_maps_training_head():
    sd = {"model_state": {"module.conv1.weight": 1, "head.1.weight": 2, "head.1.bias": 3}}
    assert clean_keys(sd) == {"conv1.weight": 1, "fc.weight": 2, "fc.bias": 3}


def test_load_and_predict_range(stand_in_weights):
    ens = Ensemble.load(stand_in_weights, device="cpu")
    p, scores = ens.predict(torch.zeros(8, 512, 512, dtype=torch.float16))
    assert len(scores) == 2 and 0.0 <= p <= 1.0
    assert abs(p - sum(scores) / 2) < 1e-9


def test_sha256_mismatch_refuses_to_load(stand_in_weights):
    with pytest.raises(WeightsError, match="SHA-256"):
        Ensemble.load(stand_in_weights, expected_sha256="0" * 64)


def test_sha256_match_loads(stand_in_weights):
    Ensemble.load(stand_in_weights, expected_sha256=sha256(stand_in_weights), device="cpu")
