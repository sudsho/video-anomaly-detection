import numpy as np

from src.transforms import ClipTransform


def test_clip_transform_shape_and_normalize():
    rng = np.random.default_rng(0)
    clip = (rng.integers(0, 255, size=(16, 64, 64, 3))).astype(np.uint8)
    tx = ClipTransform(size=112)
    x = tx(clip)
    assert tuple(x.shape) == (3, 16, 112, 112)
    # normalized values mostly within ~[-3, 3]
    assert x.float().abs().mean().item() < 5.0


def test_clip_transform_dtype():
    rng = np.random.default_rng(1)
    clip = (rng.integers(0, 255, size=(8, 32, 32, 3))).astype(np.uint8)
    tx = ClipTransform(size=64)
    x = tx(clip)
    assert x.dtype.is_floating_point
