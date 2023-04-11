import numpy as np
from batch_pystoi.stoi import stoi


def test_output_shape():
    fs = 10000

    x = np.zeros((4, 10000))
    out = stoi(x, x, fs)
    assert out.shape == (4,)

    x = np.zeros((1, 10000))
    out = stoi(x, x, fs)
    assert out.shape == (1,)

    x = np.zeros((10000))
    out = stoi(x, x, fs)
    assert out.shape == ()
