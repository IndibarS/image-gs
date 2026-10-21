"""Core checks that require neither CUDA kernels nor pretrained model downloads."""
import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from utils.quantization_utils import ste_quantize


def test_imports_do_not_load_cuda_extensions():
    result = subprocess.run(
        [sys.executable, "-c", "import main, model; import sys; "
         "assert 'fused_ssim' not in sys.modules; assert 'gsplat' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_help():
    result = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--num_gaussians" in result.stdout


@pytest.mark.parametrize("requested", ["cpu", "auto", "cuda:0"])
def test_cpu_backend_selection(monkeypatch, requested):
    model = importlib.import_module("model")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instance = model.GaussianSplatting2D.__new__(model.GaussianSplatting2D)
    torch.nn.Module.__init__(instance)
    instance._load_backend(SimpleNamespace(device=requested, topk=10))
    assert instance.device == "cpu"


def test_missing_cuda_extensions(monkeypatch):
    model = importlib.import_module("model")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setitem(sys.modules, "fused_ssim", None)
    instance = model.GaussianSplatting2D.__new__(model.GaussianSplatting2D)
    torch.nn.Module.__init__(instance)
    with pytest.warns(UserWarning):
        instance._load_backend(SimpleNamespace(device="cuda:0", topk=10))
    assert instance.device == "cpu"


def test_quantization_preserves_straight_through_gradient():
    values = torch.tensor([0.0, 0.3, 0.6, 1.0], requires_grad=True)
    result = ste_quantize(values, num_bits=2)
    torch.testing.assert_close(result, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
