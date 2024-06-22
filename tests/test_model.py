import torch

from src.model import C3D, ConvAE3D, I3DLite


def test_c3d_forward_shape():
    m = C3D(num_classes=2).eval()
    x = torch.randn(2, 3, 16, 112, 112)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 2)


def test_i3d_forward_shape():
    m = I3DLite(num_classes=2).eval()
    x = torch.randn(1, 3, 16, 112, 112)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 2)


def test_autoencoder_recon_error_shape():
    m = ConvAE3D().eval()
    x = torch.randn(3, 3, 16, 64, 64)
    with torch.no_grad():
        e = m.reconstruction_error(x)
    assert e.shape == (3,)
    assert (e >= 0).all().item()
