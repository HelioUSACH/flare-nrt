import pytest
import torch

from flare_nrt.model import Ensemble, WeightsError, clean_keys, sha256


def test_clean_keys_maps_training_head():
    sd = {"model_state": {"module.conv1.weight": 1, "head.1.weight": 2, "head.1.bias": 3}}
    assert clean_keys(sd) == {"conv1.weight": 1, "fc.weight": 2, "fc.bias": 3}


def test_unverified_weights_are_refused(stand_in_weights):
    with pytest.raises(WeightsError, match="SHA256"):
        Ensemble.load(stand_in_weights)


def test_load_and_predict_range(stand_in_weights):
    ens = Ensemble.load(stand_in_weights, sha256(stand_in_weights), device="cpu")
    p, scores = ens.predict(torch.zeros(8, 512, 512, dtype=torch.float16))
    assert len(scores) == 2 and 0.0 <= p <= 1.0
    assert abs(p - sum(scores) / 2) < 1e-9


def test_sha256_mismatch_refuses_to_load(stand_in_weights):
    with pytest.raises(WeightsError, match="SHA-256"):
        Ensemble.load(stand_in_weights, expected_sha256="0" * 64)


def test_sha256_match_loads(stand_in_weights):
    Ensemble.load(stand_in_weights, expected_sha256=sha256(stand_in_weights), device="cpu")


def test_gradcam_shape_and_range(stand_in_weights):
    ens = Ensemble.load(stand_in_weights, sha256(stand_in_weights), device="cpu")
    x = torch.rand(8, 512, 512).half()
    cam = ens.gradcam(x)
    assert cam.shape == (512, 512) and float(cam.min()) >= 0 and float(cam.max()) <= 1.0 + 1e-6
    p, _ = ens.predict(x)  # Grad-CAM leaves the network untouched
    assert abs(p - ens.predict(x)[0]) < 1e-9


def test_fetch_weights_refuses_a_wrong_file(tmp_path, monkeypatch):
    from flare_nrt import fetch_weights

    class R:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield self.body

    import requests

    monkeypatch.setattr(requests, "get", lambda url, **kw: R(b"not the weights"))
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        fetch_weights.fetch(tmp_path, "https://example.invalid")
    assert not any(tmp_path.rglob("*.pt")) and not any(tmp_path.rglob("*.part"))


def test_pinned_sha_is_the_config_default():
    from flare_nrt.config import Settings
    from flare_nrt.fetch_weights import WEIGHTS_SHA256

    assert Settings().weights_sha256 == WEIGHTS_SHA256


def test_empty_env_values_mean_defaults(monkeypatch):
    import pathlib

    from flare_nrt.config import Settings

    for line in pathlib.Path(".env.example").read_text().splitlines():
        if line.startswith("FLARE_NRT_"):
            monkeypatch.setenv(line.split("=")[0], "")
    s = Settings()
    assert s.threshold == 0.80 and s.staging_days == 4 and s.torch_threads == 2 and s.max_input_age_sec == 900
