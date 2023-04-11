import numpy as np
import pytest
from batch_pystoi.stoi import stoi


def test_different_shape_error():
    fs = 10000
    with pytest.raises(AssertionError):
        stoi(np.zeros(10), np.zeros(11), fs)
    with pytest.raises(AssertionError):
        stoi(np.zeros((10, 10)), np.zeros((10, 11)), fs)
    with pytest.raises(AssertionError):
        stoi(np.zeros(10), np.zeros((10, 11)), fs)


def test_no_1d_2d_error():
    fs = 10000
    with pytest.raises(AssertionError):
        stoi(np.array(0), np.zeros(10), fs)
    with pytest.raises(AssertionError):
        stoi(np.zeros(10), np.array(0), fs)
    with pytest.raises(AssertionError):
        stoi(np.array(0), np.array(0), fs)
    with pytest.raises(AssertionError):
        stoi(np.zeros((1, 1, 1)), np.array(10), fs)
    with pytest.raises(AssertionError):
        stoi(np.array(10), np.zeros((1, 1, 1)), fs)
    with pytest.raises(AssertionError):
        stoi(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), fs)


def test_lengths_error():
    fs = 10000
    with pytest.raises(AssertionError):
        stoi(np.zeros(10), np.zeros(10), fs, lengths=[])
    with pytest.raises(AssertionError):
        stoi(np.zeros(10), np.zeros(10), fs, lengths=[10, 10])
    with pytest.raises(AssertionError):
        stoi(np.zeros((1, 10)), np.zeros((1, 10)), fs, lengths=[10, 10])
