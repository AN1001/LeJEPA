import torch
import pytest
from lejepa.sigreg import sigreg
from lejepa.config import SigregConfig


# Apply fixed seed to all runs for reproducability
@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_sigreg_floor_for_isotropic_gaussian():
    z = torch.randn(4096, 192)
    # med 1.06, range 0.83-1.32 over 200 seeds
    assert 0.7 < sigreg(z, SigregConfig(dirs=256)).item() < 1.5


def test_sigreg_detects_wrong_scale():
    z = torch.randn(4096, 192) * 10
    # med 3370, range 3341-3406 over 200 seeds
    assert sigreg(z).item() > 3000


def test_sigreg_detects_location_and_scale_failure():
    # Sample from uniform random [0,1] which is
    # offset and has var 1/12
    z = torch.rand(4096, 192)
    # med 1990, range 1711-2375 over 200 seeds
    assert sigreg(z).item() > 1500


def test_sigreg_detects_complete_collapse():
    # All points are the same
    z = torch.randn(1, 192).repeat(4096, 1)
    # med 4342, range 3404-5689 over 200 seeds
    assert sigreg(z).item() > 3000


def test_sigreg_detects_partial_collapse():
    # Rank 1 matrix with 4096 points
    z = torch.randn(4096, 1) @ torch.randn(1, 192)
    # med 696, range 515-919 over 200 seeds
    assert sigreg(z).item() > 400
    # Rank 5 matrix with 4096 points, divide by sqrt(5) to make std 1 again
    z = torch.randn(4096, 5) @ torch.randn(5, 192) / (5**0.5)
    # med 191, range 125-272 over 200 seeds
    assert sigreg(z).item() > 100


def test_sigreg_gradients_finite():
    z = torch.randn(4096, 192, requires_grad=True)
    sigreg(z).backward()
    assert z.grad is not None
    assert z.grad.isfinite().all()
